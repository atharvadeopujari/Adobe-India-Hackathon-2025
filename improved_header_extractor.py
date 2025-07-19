"""
Improved Header Extraction with Better Filtering
Refined version that produces cleaner, more accurate results
"""

import json
import re
import os
from pathlib import Path
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


class ImprovedHeaderExtractor:
    """
    Improved header extractor with better filtering and accuracy
    """
    
    def __init__(self):
        """Initialize the extractor"""
        self.noise_patterns = [
            r'^\d+$',  # Just numbers
            r'^page\s+\d+',  # Page numbers
            r'^\d+\s*/\s*\d+$',  # Page x/y format
            r'^[^\w\s]*$',  # Only punctuation
            r'^\s*$',  # Only whitespace
            r'^https?://',  # URLs
            r'^\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)',  # Dates
            r'^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',  # Days
            r'^\d{1,2}:\d{2}',  # Times
            r'^\([^)]{1,3}\)$',  # Single characters in parentheses
            r'^[a-z]$',  # Single lowercase letters
            r'^[A-Z]$',  # Single uppercase letters
        ]
        
        self.title_skip_patterns = [
            r'^table\s+of\s+contents',
            r'^references?$',
            r'^bibliography$',
            r'^appendix',
            r'^index$',
            r'^\d+\.\d+',  # Version numbers as titles
            r'^version\s+\d+',
            r'^draft',
            r'^confidential',
            r'^internal\s+use',
        ]
        
        self.common_headers = [
            'introduction', 'background', 'overview', 'summary', 'abstract',
            'methodology', 'methods', 'approach', 'results', 'findings',
            'discussion', 'analysis', 'conclusion', 'conclusions',
            'recommendations', 'future work', 'references', 'bibliography',
            'appendix', 'acknowledgments', 'acknowledgements'
        ]
    
    def extract_document_structure(self, pdf_path: str) -> Dict:
        """
        Extract title and outline from PDF with improved filtering
        """
        try:
            if PYMUPDF_AVAILABLE:
                return self._extract_with_pymupdf_improved(pdf_path)
            elif PDFPLUMBER_AVAILABLE:
                return self._extract_with_pdfplumber_improved(pdf_path)
            else:
                return self._fallback_extraction(pdf_path)
                
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {
                "title": "Error processing document",
                "outline": []
            }
    
    def _extract_with_pymupdf_improved(self, pdf_path: str) -> Dict:
        """Extract with PyMuPDF using improved filtering"""
        doc = fitz.open(pdf_path)
        
        # Extract title
        title = self._extract_title_improved(doc)
        
        # Extract outline  
        outline = self._extract_outline_improved(doc)
        
        doc.close()
        
        return {
            "title": title,
            "outline": outline
        }
    
    def _extract_title_improved(self, doc) -> str:
        """Extract title with better filtering"""
        # Try metadata first
        metadata = doc.metadata
        if metadata.get('title') and not self._is_noise_title(metadata['title']):
            return metadata['title'].strip()
        
        # Try bookmarks
        toc = doc.get_toc()
        if toc and len(toc) > 0:
            first_bookmark = toc[0][1].strip()
            if not self._is_noise_title(first_bookmark) and len(first_bookmark) > 5:
                return first_bookmark
        
        # Look for title-like text on first page
        if len(doc) > 0:
            page = doc[0]
            blocks = page.get_text("dict")
            
            title_candidates = []
            
            for block in blocks.get("blocks", []):
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        font_size = span.get("size", 0)
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        y_pos = bbox[1]
                        
                        if (len(text) > 5 and font_size > 12 and 
                            self._is_likely_title_improved(text) and
                            not self._is_noise_title(text) and
                            y_pos < doc[0].rect.height * 0.3):  # Upper 30% of page
                            
                            title_candidates.append({
                                "text": text,
                                "size": font_size,
                                "y": y_pos,
                                "score": self._score_title_candidate(text, font_size, y_pos)
                            })
            
            if title_candidates:
                # Sort by score (descending)
                title_candidates.sort(key=lambda x: -x["score"])
                return title_candidates[0]["text"]
        
        # Fallback to filename
        filename = Path(doc.name).stem
        return self._clean_filename_title(filename)
    
    def _extract_outline_improved(self, doc) -> List[Dict]:
        """Extract outline with improved filtering"""
        outline = []
        
        # Try bookmarks/TOC first
        toc = doc.get_toc()
        if toc:
            for entry in toc:
                level, title, page = entry
                title = title.strip()
                
                if (not self._is_noise_text(title) and 
                    len(title) > 2 and len(title) < 150):
                    outline.append({
                        "level": f"H{min(level, 6)}",
                        "text": title,
                        "page": page
                    })
            
            if outline:
                return self._clean_outline(outline)
        
        # Fallback: extract headers from content
        page_headers = []
        
        for page_num in range(min(len(doc), 15)):  # Check first 15 pages
            page = doc[page_num]
            blocks = page.get_text("dict")
            
            page_text_elements = []
            
            for block in blocks.get("blocks", []):
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        font_size = span.get("size", 0)
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        
                        if text and font_size > 0:
                            page_text_elements.append({
                                "text": text,
                                "size": font_size,
                                "y": bbox[1],
                                "page": page_num + 1
                            })
            
            # Sort by y position to get reading order
            page_text_elements.sort(key=lambda x: x["y"])
            
            # Find potential headers
            avg_font_size = sum(e["size"] for e in page_text_elements) / len(page_text_elements) if page_text_elements else 12
            
            for element in page_text_elements:
                text = element["text"]
                size = element["size"]
                
                if (self._is_likely_header_improved(text, size, avg_font_size) and
                    not self._is_noise_text(text)):
                    
                    level = self._determine_header_level_improved(text, size, avg_font_size)
                    page_headers.append({
                        "level": level,
                        "text": text,
                        "page": element["page"]
                    })
        
        return self._clean_outline(page_headers)
    
    def _is_noise_text(self, text: str) -> bool:
        """Check if text is noise with improved patterns"""
        text = text.strip().lower()
        
        if len(text) < 2:
            return True
        
        for pattern in self.noise_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _is_noise_title(self, text: str) -> bool:
        """Check if text is likely noise for a title"""
        text = text.strip().lower()
        
        if len(text) < 3 or len(text) > 200:
            return True
        
        for pattern in self.title_skip_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _is_likely_title_improved(self, text: str) -> bool:
        """Improved title detection"""
        text = text.strip()
        
        if len(text) < 5 or len(text) > 150:
            return False
        
        # Good indicators for titles
        title_indicators = [
            # Mixed case with capitals
            bool(re.search(r'[A-Z]', text)) and bool(re.search(r'[a-z]', text)),
            # Contains meaningful words
            len(re.findall(r'\b[a-zA-Z]{3,}\b', text)) >= 2,
            # Not all caps (unless it's a reasonable length)
            not (text.isupper() and len(text) > 50),
            # Doesn't end with common sentence punctuation
            not text.endswith(('.', '!', '?', ';', ':')),
        ]
        
        return sum(title_indicators) >= 2
    
    def _score_title_candidate(self, text: str, font_size: float, y_pos: float) -> float:
        """Score a title candidate"""
        score = 0
        
        # Font size bonus
        if font_size > 16:
            score += 3
        elif font_size > 14:
            score += 2
        elif font_size > 12:
            score += 1
        
        # Position bonus (higher on page = better)
        if y_pos < 100:
            score += 2
        elif y_pos < 200:
            score += 1
        
        # Length bonus
        if 10 <= len(text) <= 80:
            score += 2
        elif 5 <= len(text) <= 120:
            score += 1
        
        # Content quality
        if re.search(r'\b(guide|manual|report|analysis|study|overview)\b', text, re.IGNORECASE):
            score += 1
        
        if text.count(' ') >= 2:  # Multi-word titles are better
            score += 1
        
        return score
    
    def _is_likely_header_improved(self, text: str, font_size: float, avg_font_size: float) -> bool:
        """Improved header detection"""
        text = text.strip()
        
        if len(text) < 3 or len(text) > 120:
            return False
        
        # Size-based detection
        if font_size > avg_font_size * 1.2:  # 20% larger than average
            return True
        
        # Pattern-based detection
        header_patterns = [
            r'^\d+\.\s+[A-Za-z]',  # "1. Introduction"
            r'^\d+\.\d+\s+[A-Za-z]',  # "1.1 Background"
            r'^[A-Z]\.\s+[A-Za-z]',  # "A. Methods"
            r'^(chapter|section|part)\s+\d+',  # "Chapter 1"
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Common header words
        if text.lower() in self.common_headers:
            return True
        
        # All caps headers (but not too long)
        if text.isupper() and 5 <= len(text) <= 30:
            return True
        
        return False
    
    def _determine_header_level_improved(self, text: str, font_size: float, avg_font_size: float) -> str:
        """Determine header level with improved logic"""
        text = text.strip()
        
        # Pattern-based levels
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
        size_ratio = font_size / avg_font_size if avg_font_size > 0 else 1
        
        if size_ratio > 1.5:
            return "H1"
        elif size_ratio > 1.3:
            return "H2"
        else:
            return "H3"
    
    def _clean_filename_title(self, filename: str) -> str:
        """Clean filename to make a reasonable title"""
        # Remove common file patterns
        filename = re.sub(r'^\w+\s*-\s*', '', filename)  # Remove "Microsoft Word - "
        filename = re.sub(r'\.(doc|pdf).*$', '', filename, re.IGNORECASE)
        filename = filename.replace("_", " ").replace("-", " ")
        filename = re.sub(r'\s+', ' ', filename).strip()
        
        # Title case
        if filename.islower() or filename.isupper():
            filename = filename.title()
        
        return filename if filename else "Document"
    
    def _clean_outline(self, outline: List[Dict]) -> List[Dict]:
        """Clean and filter outline entries"""
        if not outline:
            return []
        
        # Remove duplicates
        seen = set()
        cleaned = []
        
        for item in outline:
            text_key = item["text"].lower().strip()
            if text_key not in seen and len(text_key) > 2:
                seen.add(text_key)
                cleaned.append(item)
        
        # Limit to reasonable number of headers
        if len(cleaned) > 20:
            cleaned = cleaned[:20]
        
        return cleaned
    
    def _extract_with_pdfplumber_improved(self, pdf_path: str) -> Dict:
        """Extract using pdfplumber with improvements"""
        with pdfplumber.open(pdf_path) as pdf:
            title = self._extract_title_pdfplumber_improved(pdf)
            outline = self._extract_outline_pdfplumber_improved(pdf)
            
        return {
            "title": title,
            "outline": outline
        }
    
    def _extract_title_pdfplumber_improved(self, pdf) -> str:
        """Extract title using pdfplumber with improvements"""
        if len(pdf.pages) > 0:
            first_page = pdf.pages[0]
            text = first_page.extract_text()
            
            if text:
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                for line in lines[:8]:  # Check first 8 lines
                    if (self._is_likely_title_improved(line) and
                        not self._is_noise_title(line)):
                        return line
        
        # Fallback to cleaned filename
        return self._clean_filename_title(Path(pdf.path).stem)
    
    def _extract_outline_pdfplumber_improved(self, pdf) -> List[Dict]:
        """Extract outline using pdfplumber with improvements"""
        outline = []
        
        for page_num, page in enumerate(pdf.pages[:10]):
            text = page.extract_text()
            if not text:
                continue
                
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            for line in lines:
                if (self._is_likely_header_improved(line, 14, 12) and  # Assume avg font size
                    not self._is_noise_text(line)):
                    
                    level = self._determine_header_level_improved(line, 14, 12)
                    outline.append({
                        "level": level,
                        "text": line,
                        "page": page_num + 1
                    })
        
        return self._clean_outline(outline)
    
    def _fallback_extraction(self, pdf_path: str) -> Dict:
        """Fallback when no PDF library is available"""
        filename = Path(pdf_path).stem
        return {
            "title": self._clean_filename_title(filename),
            "outline": []
        }


def main():
    """Test the improved header extractor"""
    print("Improved Header Extraction Test")
    print("=" * 50)
    
    extractor = ImprovedHeaderExtractor()
    
    # Find test files
    current_dir = Path(__file__).parent
    input_dir=current_dir/"input"
    test_files = sorted(list(input_dir.glob("file*.pdf")))
    
    if not test_files:
        print("No test files found (file*.pdf)")
        return
    
    results = {}
    
    for pdf_file in test_files:
        print(f"\n📄 Processing {pdf_file.name}...")
        try:
            result = extractor.extract_document_structure(str(pdf_file))
            results[pdf_file.name] = result
            
            # Print summary
            title = result.get("title", "Unknown")
            outline_count = len(result.get("outline", []))
            print(f"   Title: {title}")
            print(f"   Headers found: {outline_count}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
            results[pdf_file.name] = {
                "title": f"Error: {str(e)}",
                "outline": []
            }
    
    # Save results
    output_file = "improved_header_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {output_file}")
    
    # Print detailed results
    print("\nDETAILED RESULTS:")
    print("=" * 50)
    
    for filename, result in results.items():
        print(f"\n📄 {filename}")
        print("-" * 30)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
