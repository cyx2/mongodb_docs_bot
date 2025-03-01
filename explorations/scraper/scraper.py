import json
import os
import re
from urllib.parse import urlparse

import scrapy
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from scrapy.crawler import CrawlerProcess

load_dotenv()


class MongoDBDocsSpider(scrapy.Spider):
    name = "mongodb_docs"
    allowed_domains = ["mongodb.com"]
    start_urls = ["https://www.mongodb.com/docs/atlas/"]

    visited_urls = set()
    scraped_urls = []

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "USER_AGENT": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
        "DOWNLOAD_DELAY": 0.1,
        "CONCURRENT_REQUESTS": 10,
        "FEED_EXPORT_ENCODING": "utf-8",
    }

    def parse(self, response):
        if response.url in self.visited_urls:
            return

        self.visited_urls.add(response.url)
        self.scraped_urls.append(response.url)
        print(f"Added new url: {response.url}")

        yield {"url": response.url, "title": response.css("title::text").get()}

        # Follow links that match our pattern
        for href in response.css("a::attr(href)"):
            url = response.urljoin(href.get())

            # Only follow URLs under the /docs/atlas/ path
            if self.should_follow(url):
                # Pass the current depth + 1 to child pages
                yield scrapy.Request(
                    url,
                    callback=self.parse,
                    meta={"depth": response.meta.get("depth", 0) + 1},
                )

    def should_follow(self, url):
        """Determine if we should follow this URL"""

        if url in self.visited_urls:
            return False

        parsed = urlparse(url)

        if parsed.netloc != "www.mongodb.com":
            return False

        if not parsed.path.startswith("/docs/atlas/"):
            return False

        # Skip anchor links (same page)
        if "#" in url:
            return False

        # Skip non-HTML files
        if re.search(r"\.(pdf|zip|png|jpg|jpeg|gif|svg)$", parsed.path):
            return False

        return True

    def closed(self, reason):
        """Called when spider is closed"""
        self.logger.info(f"Spider closed: {reason}")
        self.logger.info(f"Total URLs scraped: {len(self.scraped_urls)}")

        with open("mongodb_urls.json", "w") as f:
            json.dump(self.scraped_urls, f)


class EmbeddingCreator:
    def __init__(self):
        self._initialize_providers()
        self._initialize_mongodb()

    def _initialize_providers(self):
        self.llm = init_chat_model(
            "gemini-2.0-flash-001", model_provider="google_vertexai"
        )
        self.embeddings = VertexAIEmbeddings(model="text-embedding-004")

    def _initialize_mongodb(self):
        load_dotenv()
        mongodb_uri = os.getenv("MONGODB_URI")
        mongodb_database = os.getenv("MONGODB_DATABASE")
        prefix = "scraper_explorations"

        self.client = MongoClient(mongodb_uri)
        self.collection = self.client[mongodb_database][prefix]

        self.vector_store = MongoDBAtlasVectorSearch(
            embedding=self.embeddings,
            collection=self.collection,
            index_name=prefix,
            relevance_score_fn="cosine",
        )
        print("MongoDB connection opened. Hello!")

    def _process_url_batch(self, urls, batch_index):
        """Process a batch of URLs and create embeddings"""
        print(f"Processing batch {batch_index // 10 + 1} with {len(urls)} URLs...")

        try:
            # Use WebBaseLoader to load documents
            loader = WebBaseLoader(urls)
            docs = loader.load()

            # Print info about loaded documents
            print(f"Loaded {len(docs)} documents")

            # Split documents into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )
            all_splits = text_splitter.split_documents(docs)
            print(f"Split into {len(all_splits)} chunks")

            vector_1 = self.embeddings.embed_query(all_splits[0].page_content)
            vector_2 = self.embeddings.embed_query(all_splits[1].page_content)

            assert len(vector_1) == len(vector_2)
            print(f"Generated vectors of length {len(vector_1)}")

            _ = self.vector_store.add_documents(documents=all_splits)

            print(f"Embeddings created and saved for batch {batch_index // 10 + 1}\n")

        except Exception as e:
            print(f"Error processing batch: {e}")

    def create_embeddings_from_urls(self):
        """Load scraped URLs with WebBaseLoader and create embeddings"""
        if not os.path.exists("mongodb_urls.json"):
            print("Error: No URLs file found. Run the spider first.")
            return

        # Load URLs
        with open("mongodb_urls.json", "r") as f:
            urls = json.load(f)

        print(f"Processing {len(urls)} URLs for embeddings...")

        # Process in smaller batches to avoid memory issues
        batch_size = 10
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i : i + batch_size]
            self._process_url_batch(batch_urls, i)


def main():
    """Run the full scraper and embedding pipeline"""

    if not os.path.exists("mongodb_urls.json"):
        print("Starting scraper")
        process = CrawlerProcess(
            {
                "LOG_LEVEL": "INFO",
                "FEEDS": {
                    "mongodb_docs_output.json": {
                        "format": "json",
                        "encoding": "utf8",
                        "store_empty": False,
                        "overwrite": True,
                    }
                },
            }
        )

        process.crawl(MongoDBDocsSpider)
        process.start()

    # Then create embeddings from the collected URLs
    print("Starting embedding creator")

    embedding_creator = EmbeddingCreator()
    embedding_creator.create_embeddings_from_urls()

    print("Process completed.")


if __name__ == "__main__":
    main()
