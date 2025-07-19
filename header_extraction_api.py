"""
Header Extraction API
Standalone API for extracting titles and headers from PDF documents
"""

import sys
import os
import json
import traceback
from pathlib import Path

# Add the parent directory to sys.path to enable imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

try:
    from header_extraction_pipeline import HeaderExtractionPipeline
except ImportError as e:
    print(f"Import error: {e}")
    print("Current directory:", current_dir)
    print("Python path:", sys.path)
    raise


class HeaderExtractionAPI:
    """
    API wrapper for header extraction functionality
    """
    
    def __init__(self):
        """Initialize the API"""
        try:
            self.pipeline = HeaderExtractionPipeline()
            print("Header extraction pipeline initialized successfully")
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            traceback.print_exc()
            self.pipeline = None
    
    def extract_headers(self, pdf_path: str) -> dict:
        """
        Extract headers and title from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with 'title' and 'outline' keys
        """
        if not self.pipeline:
            return {
                "title": "Error: Pipeline not initialized",
                "outline": []
            }
        
        if not os.path.exists(pdf_path):
            return {
                "title": "Error: File not found",
                "outline": []
            }
        
        try:
            result = self.pipeline.extract_document_structure(pdf_path)
            return result
        except Exception as e:
            print(f"Error extracting headers from {pdf_path}: {e}")
            traceback.print_exc()
            return {
                "title": f"Error: {str(e)}",
                "outline": []
            }
    
    def process_all_test_files(self) -> dict:
        """
        Process all test PDF files in the current directory
        
        Returns:
            Dictionary with results for each file
        """
        results = {}
        
        # Get all PDF files starting with 'file'
        current_path = Path(__file__).parent
        test_files = list(current_path.glob("file*.pdf"))
        
        if not test_files:
            return {"error": "No test files found (file*.pdf)"}
        
        for pdf_file in sorted(test_files):
            print(f"Processing {pdf_file.name}...")
            result = self.extract_headers(str(pdf_file))
            results[pdf_file.name] = result
            
        return results


def main():
    """
    Main function to test the header extraction API
    """
    print("Header Extraction API Test")
    print("=" * 40)
    
    # Initialize API
    api = HeaderExtractionAPI()
    
    if not api.pipeline:
        print("Failed to initialize API")
        return
    
    # Process all test files
    results = api.process_all_test_files()
    
    # Print results
    print("\nRESULTS:")
    print("=" * 40)
    
    for filename, result in results.items():
        print(f"\n📄 {filename}")
        print("-" * 30)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
