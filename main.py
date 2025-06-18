import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from urllib.parse import urlparse
from firecrawl import FirecrawlApp
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import openai
import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced News Processor API v2.0",
    version="2.0.0",
    description="AI-powered news processing pipeline with multi-model orchestration, quality control, and source verification"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")

# Request/Response Models
class NewsRequest(BaseModel):
    url: Optional[str] = None
    topic: Optional[str] = None
    webhook_url: Optional[str] = None
    date_filter: Optional[str] = "24h"

class NewsResponse(BaseModel):
    success: bool
    article: str
    seo: str
    models_used: str
    processing_time: float
    input_type: str
    
    # Enhanced tracking fields
    search_query: str
    search_method: str
    num_results: int
    search_results: List[Dict[str, Any]]
    content_source: str
    raw_content: str
    extracted_facts: str
    facts_count: int
    verified_facts: str
    verification_status: str
    quality_issues: List[str]
    sources_used: List[str]
    cached: bool = False
    job_id: Optional[str] = None

class WebhookPayload(BaseModel):
    job_id: str
    status: str
    result: Optional[NewsResponse] = None
    error: Optional[str] = None
    timestamp: str

# Source credibility configuration
TRUSTED_DOMAINS = {
    "reuters.com": 9.5,
    "apnews.com": 9.5,
    "bbc.com": 9.0,
    "cnn.com": 8.0,
    "npr.org": 9.0,
    "espn.com": 8.5,
    "nfl.com": 8.5,
    "washingtonpost.com": 8.5,
    "nytimes.com": 8.5,
    "wsj.com": 9.0,
    "bloomberg.com": 8.5,
    "cbssports.com": 8.0,
    "foxsports.com": 7.5,
    "si.com": 8.0,
    "theatlantic.com": 8.5
}

BLOCKED_DOMAINS = {
    "medium.com", "reddit.com", "twitter.com", "facebook.com",
    "blog.com", "wordpress.com", "blogspot.com", "substack.com"
}

class EnhancedNewsProcessor:
    def __init__(self):
        self.openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
        self.firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY) if FIRECRAWL_API_KEY else None
        self.httpx_client = httpx.AsyncClient(timeout=30.0)
        
    def calculate_credibility_score(self, url: str) -> float:
        """Calculate credibility score for a URL"""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower().replace("www.", "")
            
            if domain in BLOCKED_DOMAINS:
                return 0.0
                
            if domain in TRUSTED_DOMAINS:
                return TRUSTED_DOMAINS[domain]
                
            # Default scoring for unknown domains
            if any(keyword in domain for keyword in ["news", "times", "post", "herald"]):
                return 7.0
            elif any(keyword in domain for keyword in ["gov", "edu", "org"]):
                return 8.0
            else:
                return 6.0
        except:
            return 0.0

    async def search_news(self, query: str, date_filter: str = "24h") -> tuple[List[Dict[str, Any]], str, str]:
        if not self.firecrawl_app:
            logger.error("Firecrawl API key not configured. Search will be skipped.")
            return [], query, "Search service not configured (Firecrawl API key missing)"

        search_method = "Firecrawl Search"

        # Convert date_filter to Firecrawl's tbs parameter
        tbs_value = None # Default to no tbs if 'none' or unmapped
        default_to_24h = False

        if date_filter == "1h":
            tbs_value = "qdr:h"
        elif date_filter == "6h":
            tbs_value = "qdr:h"  # Map 6h to past hour
            logger.info(f"Date filter '6h' mapped to Firecrawl 'qdr:h' (past hour).")
        elif date_filter == "12h":
            tbs_value = "qdr:d"  # Map 12h to past day (closer than past hour)
            logger.info(f"Date filter '12h' mapped to Firecrawl 'qdr:d' (past day).")
        elif date_filter == "24h": # This is also the Pydantic default
            tbs_value = "qdr:d"
        elif date_filter == "week":
            tbs_value = "qdr:w"
        elif date_filter == "month":
            tbs_value = "qdr:m"
        elif date_filter == "year":
            tbs_value = "qdr:y"
        elif date_filter == "none":
            tbs_value = None # Explicitly no time filter
            logger.info("Date filter 'none' selected, no time-based search filter will be applied.")
        else: # Unrecognized or None (if Pydantic default isn't hit, though it should)
            logger.warning(f"Unrecognized date_filter '{date_filter}', defaulting to '24h' (qdr:d).")
            tbs_value = "qdr:d" # Default to 24h for unrecognized values

        search_params = {
            'query': query,
            'scrape_options': {'formats': ['markdown']}, # Request Markdown content
            'limit': 7 # Fetch a bit more to filter down by credibility
        }
        if tbs_value:
            search_params['tbs'] = tbs_value

        effective_query_info = f"{query} (tbs: {tbs_value if tbs_value else 'None'})"
        logger.info(f"Performing Firecrawl search with params: {search_params}")

        try:
            # Firecrawl's search is synchronous in its current Python SDK.
            # In a real async FastAPI app, this synchronous call should be wrapped:
            # fc_search_results = await asyncio.to_thread(self.firecrawl_app.search, **search_params)
            # For this subtask, we'll show the direct call for clarity of SDK usage.
            fc_search_results = self.firecrawl_app.search(**search_params)

            results_data = []
            if isinstance(fc_search_results, dict) and 'data' in fc_search_results:
                results_data = fc_search_results['data']
            elif isinstance(fc_search_results, list): # If it directly returns a list
                results_data = fc_search_results
            else:
                if hasattr(fc_search_results, 'data'):
                    results_data = fc_search_results.data
                else:
                    logger.error(f"Unexpected Firecrawl search result format: {type(fc_search_results)}. Content: {str(fc_search_results)[:500]}")
                    return [], effective_query_info, f"Search failed: Unexpected result format"

            processed_results = []
            for res in results_data:
                url = res.get("url")
                if not url: continue

                domain = urlparse(url).netloc.lower().replace("www.", "")
                if domain in BLOCKED_DOMAINS:
                    logger.info(f"Skipping blocked domain from Firecrawl results: {domain}")
                    continue

                credibility = self.calculate_credibility_score(url)
                if credibility < 6.0: # Minimum credibility threshold
                    logger.info(f"Skipping low credibility domain: {domain} (Score: {credibility})")
                    continue

                markdown_content = res.get("markdown", "")

                processed_results.append({
                    "title": res.get("title", "No title provided"),
                    "url": url,
                    "description": res.get("description") or (markdown_content[:250] + "..." if markdown_content else "No description available."),
                    "published_at": res.get("metadata", {}).get("publishedDate") or res.get("publishedDate"),
                    "credibility_score": credibility,
                    "source_domain": domain,
                    "raw_content": markdown_content
                })

            processed_results.sort(key=lambda x: x["credibility_score"], reverse=True)
            final_results = processed_results[:5]
            logger.info(f"Found {len(final_results)} high-quality sources via Firecrawl for query: '{effective_query_info}'")
            return final_results, query, search_method
            
        except Exception as e:
            logger.error(f"Firecrawl search failed for query '{query}': {e}", exc_info=True)
            return [], query, f"Search failed: {str(e)}"

    async def extract_content(self, search_results: List[Dict[str, Any]]) -> tuple[str, str]:
        if not search_results:
            logger.warning("extract_content called with no search results.")
            return "No content could be extracted as no search results were provided.", "N/A"

        try:
            content_parts = []
            sources_used_domains = set() # Use a set to store unique domains

            for result in search_results[:3]:  # Aggregate from top 3 sources
                article_content = result.get("raw_content", "").strip()

                if not article_content:
                    logger.warning(f"No content found for source: {result.get('source_domain', 'Unknown URL')}")
                    continue

                header = (f"**Source: {result.get('source_domain', 'Unknown')} "
                          f"(Credibility: {result.get('credibility_score', 0):.1f}/10)\n"
                          f"URL: {result.get('url')}\n\n"
                          f"Title: {result.get('title', 'N/A')}\n\n")
                content_parts.append(header + article_content)

                if result.get('source_domain'):
                    sources_used_domains.add(result.get('source_domain'))
            
            if not content_parts:
                logger.warning("No content could be aggregated from search results (all were empty).")
                return "No content could be aggregated from the provided sources.", "N/A"

            aggregated_content = "\n\n---\n\n".join(content_parts)
            content_source_description = f"Aggregated content from {', '.join(sorted(list(sources_used_domains)))}" if sources_used_domains else "N/A"
            
            logger.info(f"Successfully aggregated content from {len(sources_used_domains)} unique source(s).")
            return aggregated_content, content_source_description
            
        except Exception as e:
            logger.error(f"Content aggregation failed: {e}", exc_info=True)
            return f"Content aggregation failed: {str(e)}", "Error in aggregation"

    async def extract_facts_gpt4(self, content: str, topic: str) -> tuple[str, int]:
        """Extract facts using GPT-4.1"""
        try:
            if not self.openai_client:
                return "GPT-4 not configured", 0
                
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extract key facts from news content. Focus on who, what, when, where, and why. Include source attribution."},
                    {"role": "user", "content": f"Extract facts from this content about {topic}:\n\n{content}"}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            facts = response.choices[0].message.content
            # Count facts (lines starting with bullet points or dashes)
            fact_count = len([line for line in facts.split('\n') if line.strip().startswith(('-', '•', '*'))])
            
            logger.info(f"GPT-4 extracted {fact_count} facts")
            return facts, fact_count
            
        except Exception as e:
            logger.error(f"GPT-4 fact extraction failed: {e}")
            return f"Fact extraction failed: {str(e)}", 0

    async def verify_facts_with_o3(self, facts: str, original_topic: str) -> tuple[str, str]:
        """Verify facts using the o3-2025-04-16 model."""
        if not self.openai_client:
            logger.error("OpenAI client not configured. Cannot verify facts with o3 model.")
            return "Verification skipped: OpenAI client not configured.", "SKIPPED"

        if not facts.strip():
            logger.warning("No facts provided to verify. Skipping o3 verification.")
            return "No facts provided for verification.", "SKIPPED_NO_INPUT"

        logger.info(f"Starting fact verification with o3-2025-04-16 for topic: {original_topic}")
        try:
            system_prompt = (
                "You are a meticulous fact-checker. Your role is to analyze a list of extracted facts "
                "based on provided news content about a specific topic. Assess these facts for clarity, "
                "apparent accuracy (based on general knowledge if specific context isn't available), "
                "and relevance to the stated topic. Identify any overtly questionable statements, "
                "contradictions, or areas that might require deeper scrutiny. Your output should be a "
                "structured JSON object containing two keys: 'verification_summary' (a brief textual "
                "summary of your findings) and 'verification_status' (one of: 'VERIFIED', "
                "'VERIFIED_WITH_CONCERNS', 'NEEDS_REVIEW', 'FAILED_VERIFICATION')."
            )
            user_prompt = (
                f"Topic of the news: {original_topic}\n\n"
                f"Extracted facts to verify:\n{facts}\n\n"
                "Please provide your verification analysis in the specified JSON format."
            )

            response = await self.openai_client.chat.completions.create(
                model="o3-2025-04-16",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=400, # Adjusted for potentially structured JSON output
                temperature=0.2, # Slightly higher for analytical tasks but still conservative
                response_format={"type": "json_object"} # Request JSON output
            )

            response_content = response.choices[0].message.content
            logger.info(f"o3-2025-04-16 verification API call successful. Response: {response_content[:200]}...") # Log snippet

            try:
                # Attempt to parse the JSON response from the model
                parsed_response = json.loads(response_content)
                verified_facts_summary = parsed_response.get("verification_summary", "No summary provided.")
                verification_status = parsed_response.get("verification_status", "NEEDS_REVIEW").upper()

                # Validate status
                valid_statuses = ["VERIFIED", "VERIFIED_WITH_CONCERNS", "NEEDS_REVIEW", "FAILED_VERIFICATION", "SKIPPED"]
                if verification_status not in valid_statuses:
                    logger.warning(f"o3 model returned invalid status: {verification_status}. Defaulting to NEEDS_REVIEW.")
                    verification_status = "NEEDS_REVIEW"
                    verified_facts_summary += " (Original status was invalid)"

                logger.info(f"o3 verification completed. Status: {verification_status}")
                # Return the full JSON string as verified_facts for now, or just summary
                return response_content, verification_status # Or: verified_facts_summary, verification_status

            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse JSON from o3 model: {json_err}. Response was: {response_content}")
                return "Verification result parsing error. Raw model output: " + response_content, "NEEDS_REVIEW"
            except Exception as e_parse: # Catch other parsing errors
                logger.error(f"Error processing o3 model response: {e_parse}. Response was: {response_content}")
                return "Error processing verification response. Raw model output: " + response_content, "NEEDS_REVIEW"

        except Exception as e:
            logger.error(f"o3-2025-04-16 verification failed: {e}", exc_info=True)
            return f"Verification call failed: {str(e)}", "FAILED_VERIFICATION"

    async def write_article_claude(self, verified_facts: str, topic: str) -> str:
        """Write article using Claude Sonnet 4"""
        try:
            if not self.anthropic_client:
                return f"Claude not configured. Topic: {topic}"
                
            response = await self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[
                    {"role": "user", "content": f"Write a professional news article about {topic} based on these verified facts:\n\n{verified_facts}"}
                ]
            )
            
            article = response.content[0].text
            logger.info("Claude article generation completed")
            return article
            
        except Exception as e:
            logger.error(f"Claude article generation failed: {e}")
            return f"Article generation failed: {str(e)}"

    async def generate_seo_gpt4(self, article: str, topic: str) -> str:
        """Generate SEO using GPT-4.1"""
        try:
            if not self.openai_client:
                return f"SEO optimization for {topic} (GPT-4 not configured)"
                
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Generate SEO metadata including title, description, keywords, and optimization recommendations."},
                    {"role": "user", "content": f"Generate SEO for this article about {topic}:\n\n{article[:1000]}..."}
                ],
                max_tokens=400,
                temperature=0.1
            )
            
            seo = response.choices[0].message.content
            logger.info("GPT-4 SEO generation completed")
            return seo
            
        except Exception as e:
            logger.error(f"GPT-4 SEO generation failed: {e}")
            return f"SEO generation failed: {str(e)}"

    async def _send_webhook(self, webhook_url: str, payload: WebhookPayload):
        """Send webhook notification"""
        try:
            await self.httpx_client.post(webhook_url, json=payload.dict())
            logger.info(f"Webhook sent successfully to {webhook_url}")
        except Exception as e:
            logger.error(f"Webhook failed: {e}")

    async def process_news_pipeline(self, request: NewsRequest, job_id: str = None) -> NewsResponse:
        start_time = datetime.now()
        
        if not job_id:
            job_id = hashlib.md5(f"{request.url or request.topic}{start_time.isoformat()}".encode()).hexdigest()[:12]

        # Crucial API Client checks
        if not self.openai_client:
            raise HTTPException(status_code=500, detail="OpenAI API service not configured. Cannot process request.")
        if not self.anthropic_client:
            raise HTTPException(status_code=500, detail="Anthropic API service not configured. Cannot process request.")

        raw_content = ""
        content_source_desc = ""
        search_query_used = ""
        search_method_used = ""
        search_results_list = []
        topic_for_processing = "" # Initialize topic_for_processing

        try:
            if request.url:
                input_type = "URL"
                topic_for_processing = request.url
                logger.info(f"Starting pipeline for URL: {request.url}")
                search_method_used = "Direct URL Fetch"
                search_query_used = "N/A - Direct URL"

                try:
                    async with self.httpx_client as client: # Use the class's httpx_client
                        fetch_response = await client.get(request.url, timeout=20.0)
                        fetch_response.raise_for_status()
                    # Note: fetch_response.text might be HTML. Downstream AI needs to handle it.
                    # Consider adding HTML parsing (e.g. BeautifulSoup) in a future iteration.
                    raw_content = fetch_response.text
                    parsed_url = urlparse(request.url)
                    domain_name = parsed_url.netloc.lower().replace("www.", "")
                    content_source_desc = f"Direct fetch from {domain_name}"

                    search_results_list = [{
                        "title": request.url, "url": request.url, "description": "Direct URL input",
                        "published_at": None,
                        "credibility_score": self.calculate_credibility_score(request.url),
                        "source_domain": domain_name, "raw_content": raw_content
                    }]
                    logger.info(f"Successfully fetched content from direct URL: {request.url}")

                except Exception as e:
                    logger.error(f"Failed to fetch direct URL {request.url}: {e}", exc_info=True)
                    raise HTTPException(status_code=400, detail=f"Failed to fetch content from URL '{request.url}': {str(e)}")

            elif request.topic:
                input_type = "Topic"
                topic_for_processing = request.topic
                logger.info(f"Starting pipeline for Topic: {request.topic}")

                if not self.firecrawl_app: # Check before calling search_news
                     raise HTTPException(status_code=500, detail="Firecrawl Search service not configured. Cannot process topic.")

                logger.info("Step 1: News search with Firecrawl") # Updated log message
                search_results_list, search_query_used, search_method_used = await self.search_news(request.topic, request.date_filter)

                if not search_results_list and "Search failed" not in search_method_used and "not configured" not in search_method_used : # Added check for "not configured"
                    # If search ran but found nothing (and didn't fail internally or wasn't skipped due to config)
                    logger.warning(f"No high-quality sources found by Firecrawl for topic: {request.topic}")
                    # Downstream will handle empty search_results_list

                logger.info("Step 2: Content aggregation from search results")
                raw_content, content_source_desc = await self.extract_content(search_results_list)
                if not raw_content.strip() and "failed" not in content_source_desc : # if content is empty and it wasn't an aggregation failure
                    logger.warning(f"Content aggregation yielded no textual content for topic: {request.topic}")


            else:
                raise HTTPException(status_code=400, detail="Neither URL nor Topic provided in the request.")

            MAX_CONTENT_LENGTH = 30000
            if len(raw_content) > MAX_CONTENT_LENGTH:
                logger.warning(f"Raw content length ({len(raw_content)}) exceeds limit, truncating to {MAX_CONTENT_LENGTH} chars.")
                raw_content = raw_content[:MAX_CONTENT_LENGTH]
            
            if not topic_for_processing: # Should have been set if URL or Topic was provided
                 raise HTTPException(status_code=500, detail="Internal error: Topic for processing was not set.")

            logger.info(f"Step 3: GPT-4.1 fact extraction for: {topic_for_processing}")
            extracted_facts, facts_count = await self.extract_facts_gpt4(raw_content, topic_for_processing)
            
            logger.info("Step 4: Fact verification with o3-2025-04-16")
            verified_facts, verification_status = await self.verify_facts_with_o3(extracted_facts, topic_for_processing)
            
            logger.info(f"Step 5: Article writing with claude-sonnet-4-20250514 for: {topic_for_processing}")
            article = await self.write_article_claude(verified_facts, topic_for_processing)
            
            logger.info(f"Step 6: GPT-4.1 SEO generation for: {topic_for_processing}")
            seo = await self.generate_seo_gpt4(article, topic_for_processing)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            quality_issues = []
            if input_type == "Topic" and not search_results_list and "Search failed" not in search_method_used and "not configured" not in search_method_used:
                quality_issues.append("No search results found by Firecrawl") # Updated source
            if not raw_content.strip():
                 quality_issues.append("No content was extracted or fetched for processing.")
            if facts_count < 1 and raw_content.strip(): # Only an issue if there was content
                quality_issues.append("No facts extracted despite available content.")
            elif facts_count < 3 and raw_content.strip():
                 quality_issues.append("Limited facts extracted (less than 3).")
            if verification_status != "VERIFIED":
                quality_issues.append("Fact verification indicated issues or was incomplete.")

            final_sources_used = [str(item.get('url')) for item in search_results_list if item.get('url')]

            result = NewsResponse(
                success=True, article=article, seo=seo,
                models_used="GPT-4.1 (facts + SEO), o3-2025-04-16 (verification), claude-sonnet-4-20250514 (writing)",
                processing_time=processing_time, input_type=input_type,
                search_query=search_query_used, search_method=search_method_used,
                num_results=len(search_results_list), search_results=search_results_list,
                content_source=content_source_desc,
                raw_content=raw_content[:5000] + ("..." if len(raw_content) > 5000 else ""),
                extracted_facts=extracted_facts, facts_count=facts_count,
                verified_facts=verified_facts, verification_status=verification_status,
                quality_issues=quality_issues, sources_used=final_sources_used,
                cached=False, job_id=job_id
            )
            
            logger.info(f"Pipeline completed in {processing_time:.2f}s for {input_type}: {topic_for_processing}")
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in pipeline for {request.url or request.topic}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Processing failed unexpectedly: {str(e)}")

# Initialize enhanced processor
processor = EnhancedNewsProcessor()

# Background task for async processing  
async def process_news_background(request: NewsRequest, job_id: str):
    """Background task for async news processing"""
    try:
        result = await processor.process_news_pipeline(request, job_id)
        
        # Send webhook if provided
        if request.webhook_url:
            payload = WebhookPayload(
                job_id=job_id,
                status="completed",
                result=result,
                timestamp=datetime.now().isoformat()
            )
            await processor._send_webhook(request.webhook_url, payload)
            
    except Exception as e:
        logger.error(f"Background processing failed for job {job_id}: {e}")
        
        # Send error webhook if provided
        if request.webhook_url:
            payload = WebhookPayload(
                job_id=job_id,
                status="failed",
                error=str(e),
                timestamp=datetime.now().isoformat()
            )
            await processor._send_webhook(request.webhook_url, payload)

@app.post("/news-processor", response_model=NewsResponse)
async def process_news_sync(request: NewsRequest):
    """
    Enhanced synchronous news processing endpoint.
    Now includes comprehensive source tracking and quality control.
    """
    # Validate input
    if not request.url and not request.topic:
        raise HTTPException(status_code=400, detail="Either 'url' or 'topic' must be provided")
    
    if request.url and request.topic:
        raise HTTPException(status_code=400, detail="Provide either 'url' OR 'topic', not both")
    
    try:
        return await processor.process_news_pipeline(request)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in news processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/news-processor/async")
async def process_news_async(request: NewsRequest, background_tasks: BackgroundTasks):
    """
    Enhanced asynchronous news processing endpoint.
    """
    # Validate input
    if not request.url and not request.topic:
        raise HTTPException(status_code=400, detail="Either 'url' or 'topic' must be provided")
    
    if request.url and request.topic:
        raise HTTPException(status_code=400, detail="Provide either 'url' OR 'topic', not both")
    
    if not request.webhook_url:
        raise HTTPException(status_code=400, detail="webhook_url is required for async processing")
    
    # Generate job ID
    job_id = hashlib.md5(f"{request.url or request.topic}{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    
    # Start background processing
    background_tasks.add_task(process_news_background, request, job_id)
    
    return {
        "success": True,
        "job_id": job_id,
        "status": "processing",
        "message": "Job started. Result will be sent to webhook_url when complete.",
        "webhook_url": request.webhook_url
    }

@app.post("/webhook/news-processor", response_model=NewsResponse)
async def webhook_endpoint(request: Request):
    """
    Webhook endpoint for external systems to trigger news processing.
    """
    try:
        body = await request.json()
        news_request = NewsRequest(**body)
        return await process_news_sync(news_request)
        
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid webhook payload: {str(e)}")

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Check status of async job.
    """
    return {
        "job_id": job_id,
        "status": "unknown",
        "message": "Job tracking not implemented. Use webhook for async results."
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.head("/")
async def root_head():
    """HEAD method for root endpoint"""
    return {}

@app.head("/health")
async def health_head():
    """HEAD method for health endpoint"""
    return {}

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Enhanced News Processor API",
        "version": "2.0.0",
        "endpoints": {
            "sync_process": "/news-processor",
            "async_process": "/news-processor/async", 
            "webhook": "/webhook/news-processor",
            "job_status": "/job/{job_id}",
            "health": "/health",
            "docs": "/docs"
        },
        "features": [
            "Enhanced multi-source pipeline",
            "Quality-controlled source filtering",
            "Comprehensive fact verification", 
            "Source attribution and tracking",
            "Detailed processing transparency",
            "Quality gates and issue detection"
        ],
        "new_in_v2": [
            "Multi-source content aggregation",
            "Source credibility scoring",
            "Enhanced fact verification", 
            "Quality issue detection",
            "Detailed pipeline tracking",
            "Date filtering for recent content"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
