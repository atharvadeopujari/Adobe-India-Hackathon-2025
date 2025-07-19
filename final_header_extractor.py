"""
Final Header Extraction Module
Production-ready module that extracts document titles and hierarchical headers
from PDF files, returning the exact JSON format specified.
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import PDF processing libraries
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


class PDFHeaderExtractor:
    """
    Main class for extracting document structure from PDFs
    Returns JSON in the exact format: {"title": "...", "outline": [...]}
    """
    
    def __init__(self):
        """Initialize the extractor with filtering patterns"""
        self.noise_patterns = [
            r'^\d+$',  # Just numbers
            r'^page\s+\d+',  # Page numbers  
            r'^\d+\s*/\s*\d+$',  # Page x/y
            r'^[^\w\s]*$',  # Only punctuation
            r'^https?://',  # URLs
            r'^\d{1,2}:\d{2}',  # Times
            r'^\([^)]{1,3}\)$',  # Single chars in parentheses
        ]
        
        self.header_keywords = [
            'introduction', 'background', 'overview', 'summary', 'abstract',
            'methodology', 'methods', 'approach', 'results', 'findings', 
            'discussion', 'analysis', 'conclusion', 'conclusions',
            'recommendations', 'references', 'appendix', 'acknowledgments'
        ]
    
    def extract_structure(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract document structure from PDF
        
        Args:
            pdf_path (str): Path to PDF file
            
        Returns:
            Dict: {"title": str, "outline": List[Dict]}
                  where outline items have "level", "text", and "page" keys
        """
        if not os.path.exists(pdf_path):
            return {"title": "File not found", "outline": []}
        
        try:
            # Use the best available library
            if HAS_PYMUPDF:
                return self._extract_with_pymupdf(pdf_path)
            elif HAS_PDFPLUMBER:
                return self._extract_with_pdfplumber(pdf_path)
            elif HAS_PYPDF:
                return self._extract_with_pypdf(pdf_path)
            else:
                logger.warning("No PDF libraries available")
                return self._fallback_extract(pdf_path)
                
        except Exception as e:
            logger.error(f"Error extracting from {pdf_path}: {e}")
            return {"title": "Error processing document", "outline": []}
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Dict[str, any]:
        """Extract using PyMuPDF (recommended)"""
        doc = fitz.open(pdf_path)
        
        try:
            # Extract title
            title = self._get_title_pymupdf(doc, pdf_path)
            
            # Extract outline/headers
            outline = self._get_outline_pymupdf(doc)
            
            return {"title": title, "outline": outline}
            
        finally:
            doc.close()
    
    def _get_title_pymupdf(self, doc, pdf_path: str) -> str:
        """Get document title using PyMuPDF"""
        # Strategy 1: PDF metadata
        metadata = doc.metadata
        if metadata.get('title'):
            title = metadata['title'].strip()
            if self._is_valid_title(title):
                return title
        
        # Strategy 2: TOC/bookmarks 
        toc = doc.get_toc()
        if toc and len(toc) > 0:
            first_entry = toc[0][1].strip()
            if self._is_valid_title(first_entry) and len(first_entry) > 8:
                return first_entry
        
        # Strategy 3: Large text on first page
        if len(doc) > 0:
            page = doc[0]
            blocks = page.get_text("dict")
            
            candidates = []
            page_height = page.rect.height
            
            for block in blocks.get("blocks", []):
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        size = span.get("size", 0)
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        y_pos = bbox[1]
                        
                        # Look for large text in upper portion of page
                        if (size > 12 and 
                            y_pos < page_height * 0.4 and
                            self._is_valid_title(text) and
                            8 <= len(text) <= 150):
                            
                            candidates.append({
                                "text": text,
                                "size": size, 
                                "y": y_pos,
                                "score": self._score_title(text, size, y_pos)
                            })
            
            if candidates:
                candidates.sort(key=lambda x: -x["score"])
                return candidates[0]["text"]
        
        # Fallback: cleaned filename
        return self._clean_filename(pdf_path)
    
    def _get_outline_pymupdf(self, doc) -> List[Dict]:
        """Get document outline using PyMuPDF"""
        outline = []
        
        # Strategy 1: Use PDF bookmarks/TOC
        toc = doc.get_toc()
        if toc:
            for entry in toc:
                level, title, page = entry
                title = title.strip()
                
                if self._is_valid_header(title):
                    outline.append({
                        "level": f"H{min(level, 6)}",
                        "text": title,
                        "page": page
                    })
            
            # If we got good bookmarks, return them
            if len(outline) >= 2:
                return outline[:15]  # Limit to 15 entries
        
        # Strategy 2: Extract headers from content 
        return self._extract_content_headers_pymupdf(doc)
    
    def _extract_content_headers_pymupdf(self, doc) -> List[Dict]:
        """Extract headers from document content"""
        headers = []
        
        # Analyze first several pages
        for page_num in range(min(len(doc), 10)):
            page = doc[page_num]
            blocks = page.get_text("dict")
            
            text_elements = []
            
            # Collect text elements with font info
            for block in blocks.get("blocks", []):
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        size = span.get("size", 0)
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        
                        if text and size > 0:
                            text_elements.append({
                                "text": text,
                                "size": size,
                                "y": bbox[1],
                                "page": page_num + 1
                            })
            
            # Sort by reading order (y position)
            text_elements.sort(key=lambda x: x["y"])
            
            # Calculate average font size for this page
            if text_elements:
                avg_size = sum(e["size"] for e in text_elements) / len(text_elements)
                
                # Find potential headers
                for elem in text_elements:
                    text = elem["text"] 
                    size = elem["size"]
                    
                    if self._is_header_candidate(text, size, avg_size):
                        level = self._determine_level(text, size, avg_size)
                        headers.append({
                            "level": level,
                            "text": text,
                            "page": elem["page"]
                        })
        
        return self._filter_headers(headers)
    
    def _is_valid_title(self, text: str) -> bool:
        """Check if text could be a valid document title"""
        text = text.strip()
        
        # Length check
        if len(text) < 3 or len(text) > 200:
            return False
            
        # Skip obvious non-titles
        skip_patterns = [
            r'^table\s+of\s+contents',
            r'^references?$', 
            r'^bibliography$',
            r'^\d+\.\d+',  # Version numbers
            r'^draft\s*$',
            r'^page\s+\d+',
            r'^\d+$'
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, text.lower()):
                return False
        
        return True
    
    def _is_valid_header(self, text: str) -> bool:
        """Check if text could be a valid header"""
        text = text.strip()
        
        if len(text) < 2 or len(text) > 150:
            return False
            
        # Check against noise patterns
        for pattern in self.noise_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False
        
        return True
    
    def _score_title(self, text: str, font_size: float, y_pos: float) -> float:
        """Score a title candidate"""
        score = 0.0
        
        # Font size scoring
        if font_size >= 18:
            score += 4
        elif font_size >= 16:
            score += 3
        elif font_size >= 14:
            score += 2
        elif font_size >= 12:
            score += 1
        
        # Position scoring (higher on page = better)
        if y_pos < 80:
            score += 3
        elif y_pos < 150:
            score += 2
        elif y_pos < 250:
            score += 1
        
        # Length scoring
        if 15 <= len(text) <= 80:
            score += 2
        elif 8 <= len(text) <= 120:
            score += 1
            
        # Content scoring
        if any(word in text.lower() for word in ['report', 'analysis', 'guide', 'manual', 'study']):
            score += 1
        
        if text.count(' ') >= 2:  # Multi-word titles
            score += 1
            
        return score
    
    def _is_header_candidate(self, text: str, font_size: float, avg_size: float) -> bool:
        """Check if text is a potential header"""
        text = text.strip()
        
        if len(text) < 3 or len(text) > 100:
            return False
            
        if not self._is_valid_header(text):
            return False
        
        # Size-based detection (larger than average)
        if font_size > avg_size * 1.15:
            return True
        
        # Pattern-based detection
        patterns = [
            r'^\d+\.\s+[A-Za-z]',  # "1. Introduction"
            r'^\d+\.\d+\s+[A-Za-z]',  # "1.1 Background" 
            r'^[A-Z]\.\s+[A-Za-z]',  # "A. Methods"
            r'^(chapter|section|part)\s+\d+',  # "Chapter 1"
        ]
        
        for pattern in patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Keyword-based detection
        if text.lower() in self.header_keywords:
            return True
        
        # All-caps headers (reasonable length)
        if text.isupper() and 4 <= len(text) <= 25:
            return True
        
        return False
    
    def _determine_level(self, text: str, font_size: float, avg_size: float) -> str:
        """Determine header level H1-H6"""
        # Pattern-based levels first
        if re.match(r'^\d+\.\s+', text):
            return "H1"
        elif re.match(r'^\d+\.\d+\s+', text):
            return "H2" 
        elif re.match(r'^\d+\.\d+\.\d+\s+', text):
            return "H3"
        elif re.match(r'^(chapter|part)\s+\d+', text, re.IGNORECASE):
            return "H1"
        elif re.match(r'^(section)\s+\d+', text, re.IGNORECASE):
            return "H2"
        
        # Size-based levels
        ratio = font_size / avg_size if avg_size > 0 else 1
        
        if ratio >= 1.5:
            return "H1"
        elif ratio >= 1.3:
            return "H2" 
        elif ratio >= 1.2:
            return "H3"
        else:
            return "H2"  # Default
    
    def _filter_headers(self, headers: List[Dict]) -> List[Dict]:
        """Filter and clean header list"""
        if not headers:
            return []
        
        # Remove duplicates by text
        seen = set()
        filtered = []
        
        for header in headers:
            text_key = header["text"].lower().strip()
            if text_key not in seen:
                seen.add(text_key)
                filtered.append(header)
        
        # Sort by page and limit count
        filtered.sort(key=lambda x: (x["page"], x.get("order", 0)))
        
        return filtered[:12]  # Limit to 12 headers max
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> Dict[str, any]:
        """Extract using pdfplumber as fallback"""
        with pdfplumber.open(pdf_path) as pdf:
            title = self._get_title_pdfplumber(pdf, pdf_path)
            outline = self._get_outline_pdfplumber(pdf)
            
            return {"title": title, "outline": outline}
    
    def _get_title_pdfplumber(self, pdf, pdf_path: str) -> str:
        """Get title using pdfplumber"""
        if pdf.pages:
            first_page = pdf.pages[0]
            text = first_page.extract_text()
            
            if text:
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                for line in lines[:6]:
                    if self._is_valid_title(line) and 8 <= len(line) <= 120:
                        return line
        
        return self._clean_filename(pdf_path)
    
    def _get_outline_pdfplumber(self, pdf) -> List[Dict]:
        """Get outline using pdfplumber"""
        headers = []
        
        for page_num, page in enumerate(pdf.pages[:8]):  # Check first 8 pages
            text = page.extract_text()
            if not text:
                continue
                
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            for line in lines:
                if (self._is_header_candidate(line, 14, 12) and  # Assume font sizes
                    self._is_valid_header(line)):
                    
                    level = self._determine_level(line, 14, 12)
                    headers.append({
                        "level": level,
                        "text": line,
                        "page": page_num + 1
                    })
        
        return self._filter_headers(headers)
    
    def _extract_with_pypdf(self, pdf_path: str) -> Dict[str, any]:
        """Extract using pypdf as fallback"""
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            
            title = self._get_title_pypdf(reader, pdf_path)
            outline = self._get_outline_pypdf(reader)
            
            return {"title": title, "outline": outline}
    
    def _get_title_pypdf(self, reader, pdf_path: str) -> str:
        """Get title using pypdf"""
        # Try metadata
        if reader.metadata and reader.metadata.get('/Title'):
            title = reader.metadata['/Title'].strip()
            if self._is_valid_title(title):
                return title
        
        # Try first page content
        if reader.pages:
            text = reader.pages[0].extract_text()
            if text:
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                for line in lines[:6]:
                    if self._is_valid_title(line):
                        return line
        
        return self._clean_filename(pdf_path)
    
    def _get_outline_pypdf(self, reader) -> List[Dict]:
        """Get outline using pypdf"""
        headers = []
        
        # Try bookmarks
        if reader.outline:
            def process_outline_items(items, level=1):
                for item in items:
                    if hasattr(item, 'title'):
                        text = item.title.strip()
                        if self._is_valid_header(text):
                            headers.append({
                                "level": f"H{min(level, 6)}",
                                "text": text,
                                "page": 1  # pypdf doesn't easily give page numbers
                            })
                    elif isinstance(item, list):
                        process_outline_items(item, level + 1)
            
            process_outline_items(reader.outline)
            
            if headers:
                return self._filter_headers(headers)
        
        # Fallback: extract from content
        for page_num, page in enumerate(reader.pages[:8]):
            text = page.extract_text()
            if not text:
                continue
                
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            for line in lines:
                if (self._is_header_candidate(line, 14, 12) and
                    self._is_valid_header(line)):
                    
                    level = self._determine_level(line, 14, 12)
                    headers.append({
                        "level": level,
                        "text": line,
                        "page": page_num + 1
                    })
        
        return self._filter_headers(headers)
    
    def _clean_filename(self, pdf_path: str) -> str:
        """Clean filename to make reasonable title"""
        filename = Path(pdf_path).stem
        
        # Remove common prefixes
        filename = re.sub(r'^(microsoft\s+word\s*-\s*)', '', filename, flags=re.IGNORECASE)
        filename = re.sub(r'\.(doc|pdf).*$', '', filename, flags=re.IGNORECASE)
        
        # Clean up
        filename = filename.replace('_', ' ').replace('-', ' ')
        filename = re.sub(r'\s+', ' ', filename).strip()
        
        # Title case if all lower or upper
        if filename.islower() or filename.isupper():
            filename = filename.title()
        
        return filename if filename else "Document"
    
    def _fallback_extract(self, pdf_path: str) -> Dict[str, any]:
        """Fallback when no libraries available"""
        return {
            "title": self._clean_filename(pdf_path),
            "outline": []
        }


# Convenience function for direct usage
def extract_pdf_headers(pdf_path: str) -> Dict[str, any]:
    """
    Extract title and headers from a PDF file
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        Dict: {"title": str, "outline": [{"level": str, "text": str, "page": int}]}
    """
    extractor = PDFHeaderExtractor()
    return extractor.extract_structure(pdf_path)


# Test function
def test_extraction():
    """Test the extraction on all PDF files"""
    print("PDF Header Extraction - Final Test")
    print("=" * 50)
    
    # Check what libraries are available
    libs = []
    if HAS_PYMUPDF:
        libs.append("PyMuPDF")
    if HAS_PDFPLUMBER:  
        libs.append("pdfplumber")
    if HAS_PYPDF:
        libs.append("pypdf")
    
    print(f"Available libraries: {', '.join(libs) if libs else 'None'}")
    print()
    
    # Find test files
    current_dir = Path(".")
    test_files = sorted(list(current_dir.glob("file*.pdf")))
    
    if not test_files:
        print("No test files found matching pattern 'file*.pdf'")
        return
    
    extractor = PDFHeaderExtractor()
    results = {}
    
    for pdf_file in test_files:
        print(f"📄 Processing {pdf_file.name}...")
        
        try:
            result = extractor.extract_structure(str(pdf_file))
            results[pdf_file.name] = result
            
            title = result["title"]
            header_count = len(result["outline"])
            
            print(f"   ✅ Title: {title}")
            print(f"   📑 Headers: {header_count}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[pdf_file.name] = {"title": f"Error: {e}", "outline": []}
    
    # Save results
    output_file = "final_extraction_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {output_file}")
    
    # Show sample output
    print(f"\n📋 SAMPLE OUTPUT:")
    print("-" * 30)
    
    if results:
        sample_key = next(iter(results))
        sample_result = results[sample_key]
        print(f"File: {sample_key}")
        print(json.dumps(sample_result, indent=2, ensure_ascii=False))
    
    return results


if __name__ == "__main__":
    test_extraction()
