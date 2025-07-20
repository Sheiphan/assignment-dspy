#!/usr/bin/env python3
"""
Modular Text-to-Structured Data Pipeline Prototype

This prototype demonstrates key concepts:
- Schema-first design with Pydantic
- Confidence scoring
- Validation and error handling
- Extensible architecture

Usage:
    python prototype_pipeline.py
"""

import json
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
from abc import ABC, abstractmethod
import re
import hashlib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Schema Definitions (Customize for your use case)
# =============================================================================

class ContactInfo(BaseModel):
    """Nested contact information structure"""
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    
    @validator('email')
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v

class CompanyInfo(BaseModel):
    """Company information with nested structure"""
    name: str
    industry: Optional[str] = None
    size: Optional[str] = None
    location: Optional[str] = None
    contact: Optional[ContactInfo] = None
    
class PersonInfo(BaseModel):
    """Person information"""
    name: str
    title: Optional[str] = None
    experience_years: Optional[int] = None
    skills: List[str] = Field(default_factory=list)

class ExtractedData(BaseModel):
    """Main output schema - customize this for your domain"""
    document_type: str
    companies: List[CompanyInfo] = Field(default_factory=list)
    people: List[PersonInfo] = Field(default_factory=list)
    key_topics: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# =============================================================================
# Core Pipeline Components
# =============================================================================

@dataclass
class ExtractionResult:
    """Container for extraction results with metadata"""
    data: ExtractedData
    confidence: float
    processing_time: float
    tokens_used: int
    errors: List[str] = None

class LLMInterface(ABC):
    """Abstract interface for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        pass

class MockLLM(LLMInterface):
    """Mock LLM for testing - replace with real implementation"""
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Simulate realistic JSON output based on schema
        mock_response = {
            "document_type": "business_profile",
            "companies": [
                {
                    "name": "TechCorp Inc",
                    "industry": "Technology",
                    "size": "500-1000 employees",
                    "location": "San Francisco, CA",
                    "contact": {
                        "email": "info@techcorp.com",
                        "phone": "+1-555-0123"
                    }
                }
            ],
            "people": [
                {
                    "name": "John Smith",
                    "title": "CEO",
                    "experience_years": 15,
                    "skills": ["leadership", "strategy", "technology"]
                }
            ],
            "key_topics": ["artificial intelligence", "machine learning", "cloud computing"],
            "summary": "Technology company focused on AI solutions"
        }
        return json.dumps(mock_response, indent=2)
    
    def get_token_count(self, text: str) -> int:
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4

class ConfidenceEstimator:
    """Estimates confidence scores for extracted data"""
    
    def estimate_field_confidence(self, field_name: str, value: Any, context: str) -> float:
        """Estimate confidence for a specific field"""
        if value is None or (isinstance(value, (list, dict, str)) and not value):
            return 0.0
        
        # Simple heuristics - replace with more sophisticated methods
        confidence = 0.5  # Base confidence
        
        # Increase confidence for structured data
        if isinstance(value, dict) and len(value) > 1:
            confidence += 0.2
        
        # Increase confidence for specific patterns
        if field_name == "email" and "@" in str(value):
            confidence += 0.3
        
        if field_name == "phone" and re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', str(value)):
            confidence += 0.3
            
        return min(confidence, 1.0)
    
    def estimate_overall_confidence(self, data: ExtractedData) -> float:
        """Estimate overall confidence for the extracted data"""
        field_confidences = []
        
        # Calculate confidence for each field
        for field_name, field_value in data.dict().items():
            if field_name not in ['confidence_scores', 'metadata']:
                conf = self.estimate_field_confidence(field_name, field_value, "")
                field_confidences.append(conf)
        
        return sum(field_confidences) / len(field_confidences) if field_confidences else 0.0

class DataValidator:
    """Validates extracted data against schema and business rules"""
    
    def validate(self, data: ExtractedData) -> List[str]:
        """Validate data and return list of errors"""
        errors = []
        
        # Schema validation is handled by Pydantic automatically
        
        # Business rule validation
        if not data.companies and not data.people:
            errors.append("No companies or people found in document")
        
        for company in data.companies:
            if company.contact and company.contact.email:
                if '@' not in company.contact.email:
                    errors.append(f"Invalid email format: {company.contact.email}")
        
        return errors

class TextToStructuredPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.confidence_estimator = ConfidenceEstimator()
        self.validator = DataValidator()
        self.extraction_cache = {}
    
    def _build_extraction_prompt(self, text: str, schema: Dict) -> str:
        """Build the extraction prompt with schema guidance"""
        schema_str = json.dumps(schema, indent=2)
        
        prompt = f"""
Extract structured information from the following text and return it as valid JSON matching this schema:

SCHEMA:
{schema_str}

INSTRUCTIONS:
1. Extract only information that is explicitly mentioned in the text
2. Use null for missing fields rather than making assumptions
3. For lists, include all relevant items found
4. Ensure the output is valid JSON
5. Include confidence indicators where uncertain

TEXT TO EXTRACT FROM:
{text}

EXTRACTED JSON:
"""
        return prompt
    
    def _parse_llm_output(self, output: str) -> Dict:
        """Parse and clean LLM output"""
        # Remove any markdown formatting
        output = re.sub(r'```json\n?', '', output)
        output = re.sub(r'```\n?', '', output)
        
        try:
            return json.loads(output.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Raw output: {output}")
            raise ValueError(f"Invalid JSON output from LLM: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for input text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def extract(self, text: str, use_cache: bool = True) -> ExtractionResult:
        """
        Main extraction method
        
        Args:
            text: Input text to extract data from
            use_cache: Whether to use caching for repeated extractions
            
        Returns:
            ExtractionResult containing extracted data and metadata
        """
        start_time = datetime.now()
        cache_key = self._get_cache_key(text) if use_cache else None
        
        # Check cache
        if use_cache and cache_key in self.extraction_cache:
            logger.info("Using cached result")
            return self.extraction_cache[cache_key]
        
        try:
            # Build prompt with schema
            schema = ExtractedData.schema()
            prompt = self._build_extraction_prompt(text, schema)
            
            # Generate with LLM
            logger.info("Calling LLM for extraction...")
            raw_output = self.llm.generate(prompt)
            tokens_used = self.llm.get_token_count(prompt + raw_output)
            
            # Parse output
            parsed_data = self._parse_llm_output(raw_output)
            
            # Validate and create structured object
            extracted_data = ExtractedData(**parsed_data)
            
            # Validate business rules
            validation_errors = self.validator.validate(extracted_data)
            
            # Estimate confidence
            overall_confidence = self.confidence_estimator.estimate_overall_confidence(extracted_data)
            
            # Add confidence scores to the data
            extracted_data.confidence_scores["overall"] = overall_confidence
            extracted_data.metadata = {
                "extraction_timestamp": datetime.now().isoformat(),
                "tokens_used": tokens_used,
                "validation_errors": validation_errors
            }
            
            # Create result
            processing_time = (datetime.now() - start_time).total_seconds()
            result = ExtractionResult(
                data=extracted_data,
                confidence=overall_confidence,
                processing_time=processing_time,
                tokens_used=tokens_used,
                errors=validation_errors
            )
            
            # Cache result
            if use_cache:
                self.extraction_cache[cache_key] = result
            
            logger.info(f"Extraction completed in {processing_time:.2f}s with confidence {overall_confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            # Return empty result with error
            processing_time = (datetime.now() - start_time).total_seconds()
            return ExtractionResult(
                data=ExtractedData(document_type="unknown"),
                confidence=0.0,
                processing_time=processing_time,
                tokens_used=0,
                errors=[str(e)]
            )

# =============================================================================
# Usage Example and Testing
# =============================================================================

def main():
    """Example usage of the pipeline"""
    
    # Sample text to extract from
    sample_text = """
    TechCorp Inc is a growing technology company based in San Francisco, California. 
    Founded in 2018, the company specializes in artificial intelligence and machine learning solutions.
    
    The company employs between 500-1000 people and is led by CEO John Smith, who has over 15 years 
    of experience in the technology sector. John previously worked at major tech companies and is 
    known for his expertise in leadership, strategy, and emerging technologies.
    
    For business inquiries, TechCorp can be reached at info@techcorp.com or by phone at +1-555-0123.
    The company's main office is located at 123 Innovation Drive, San Francisco, CA 94105.
    
    TechCorp's main focus areas include cloud computing, artificial intelligence, and machine learning.
    They are currently working on several innovative projects in the AI space.
    """
    
    # Initialize pipeline
    llm = MockLLM()  # Replace with real LLM implementation
    pipeline = TextToStructuredPipeline(llm)
    
    # Extract data
    print("Running extraction pipeline...")
    result = pipeline.extract(sample_text)
    
    # Display results
    print(f"\n{'='*50}")
    print("EXTRACTION RESULTS")
    print(f"{'='*50}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    print(f"Tokens Used: {result.tokens_used}")
    
    if result.errors:
        print(f"\nErrors: {result.errors}")
    
    print(f"\nExtracted Data:")
    print(json.dumps(result.data.dict(), indent=2))
    
    # Demonstrate schema validation
    print(f"\n{'='*50}")
    print("SCHEMA VALIDATION DEMO")
    print(f"{'='*50}")
    
    try:
        # This should work
        valid_data = ExtractedData(
            document_type="test",
            companies=[CompanyInfo(name="Test Corp")]
        )
        print("✅ Valid data created successfully")
        
        # This should fail
        invalid_data = ExtractedData(
            document_type="test",
            companies=[CompanyInfo(
                name="Test Corp",
                contact=ContactInfo(email="invalid-email")  # Missing @
            )]
        )
        print("❌ This should have failed validation!")
        
    except Exception as e:
        print(f"✅ Validation caught error as expected: {e}")

if __name__ == "__main__":
    main() 