"""
Header Extraction Pipeline for PDF Documents
Uses RAGflow deepdoc functionality to extract document titles and hierarchical headers
"""

import json
import re
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from .parser.pdf_parser import RAGFlowPdfParser
from .vision import LayoutRecognizer, OCR


class HeaderExtractionPipeline:
    """
    Pipeline to extract document titles and hierarchical headers from PDF files
    """
    
    def __init__(self):
        """Initialize the header extraction pipeline"""
        self.pdf_parser = RAGFlowPdfParser()
        
    def extract_document_structure(self, pdf_path: str) -> Dict:
        """
        Main method to extract title and outline from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict with 'title' and 'outline' keys
        """
        try:
            # Parse the PDF document
            boxes, tables = self.pdf_parser(pdf_path, need_image=False)
            
            # Extract title
            title = self._extract_title(boxes, pdf_path)
            
            # Extract hierarchical headers
            outline = self._extract_outline(boxes, pdf_path)
            
            return {
                "title": title,
                "outline": outline
            }
            
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {str(e)}")
            return {
                "title": "Unknown Document",
                "outline": []
            }
    
    def _extract_title(self, boxes: List[Dict], pdf_path: str) -> str:
        """
        Extract document title from the parsed boxes
        
        Args:
            boxes: List of text boxes from PDF parsing
            pdf_path: Path to PDF file
            
        Returns:
            Document title as string
        """
        title_candidates = []
        
        # Strategy 1: Look for boxes marked as "title" layout type
        for box in boxes:
            if box.get("layout_type") == "title" and box.get("page_number", 1) <= 2:
                title_candidates.append({
                    "text": box["text"].strip(),
                    "page": box.get("page_number", 1),
                    "confidence": 0.9,
                    "y_pos": box.get("top", 0)
                })
        
        # Strategy 2: Look for large font text at the top of first pages
        if not title_candidates:
            first_page_boxes = [box for box in boxes if box.get("page_number", 1) <= 2]
            if first_page_boxes:
                # Sort by page then by y position
                first_page_boxes.sort(key=lambda x: (x.get("page_number", 1), x.get("top", 0)))
                
                for i, box in enumerate(first_page_boxes[:10]):  # Check first 10 boxes
                    text = box["text"].strip()
                    if self._is_likely_title(text):
                        title_candidates.append({
                            "text": text,
                            "page": box.get("page_number", 1),
                            "confidence": 0.7 - i * 0.05,  # Decreasing confidence
                            "y_pos": box.get("top", 0)
                        })
        
        # Strategy 3: Check PDF metadata/bookmarks
        if hasattr(self.pdf_parser, 'outlines') and self.pdf_parser.outlines:
            first_outline = self.pdf_parser.outlines[0]
            if isinstance(first_outline, tuple) and len(first_outline) >= 2:
                title_candidates.append({
                    "text": first_outline[0].strip(),
                    "page": 1,
                    "confidence": 0.8,
                    "y_pos": 0
                })
        
        # Select best title candidate
        if title_candidates:
            # Sort by confidence, then by page, then by y position
            title_candidates.sort(key=lambda x: (-x["confidence"], x["page"], x["y_pos"]))
            return title_candidates[0]["text"]
        
        # Fallback: Use filename
        import os
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        return filename.replace("_", " ").replace("-", " ").title()
    
    def _is_likely_title(self, text: str) -> bool:
        """
        Check if text is likely to be a document title
        
        Args:
            text: Text to evaluate
            
        Returns:
            True if text appears to be a title
        """
        text = text.strip()
        
        # Skip if too short or too long
        if len(text) < 3 or len(text) > 200:
            return False
        
        # Skip if it's just numbers, dates, or common headers/footers
        if re.match(r'^[\d\s\-/\.]+$', text):
            return False
        
        # Skip common non-title patterns
        skip_patterns = [
            r'^\d+$',  # Just numbers
            r'^page\s+\d+',  # Page numbers
            r'^\d+\s*/\s*\d+$',  # Page x/y format
            r'^table\s+of\s+contents',  # TOC
            r'^references?$',  # References
            r'^bibliography$',  # Bibliography
            r'^appendix',  # Appendix
            r'^abstract$',  # Abstract
            r'^introduction$',  # Introduction (when alone)
        ]
        
        for pattern in skip_patterns:
            if re.match(pattern, text.lower()):
                return False
        
        # Prefer text with mixed case and reasonable length
        if 10 <= len(text) <= 100:
            return True
        
        return len(text) > 5
    
    def _extract_outline(self, boxes: List[Dict], pdf_path: str) -> List[Dict]:
        """
        Extract hierarchical outline/headers from the document
        
        Args:
            boxes: List of text boxes from PDF parsing
            pdf_path: Path to PDF file
            
        Returns:
            List of outline items with level, text, and page
        """
        outline = []
        
        # Strategy 1: Use PDF bookmarks if available
        if hasattr(self.pdf_parser, 'outlines') and self.pdf_parser.outlines:
            for outline_item in self.pdf_parser.outlines:
                if isinstance(outline_item, tuple) and len(outline_item) >= 2:
                    title, depth = outline_item
                    level = f"H{min(depth + 1, 6)}"  # HTML-style heading levels
                    outline.append({
                        "level": level,
                        "text": title.strip(),
                        "page": 1  # We don't have page info from bookmarks
                    })
        
        # Strategy 2: Extract from layout-recognized headers
        if not outline:
            outline = self._extract_headers_from_layout(boxes)
        
        # Strategy 3: Pattern-based header detection
        if not outline:
            outline = self._extract_headers_from_patterns(boxes)
        
        return outline
    
    def _extract_headers_from_layout(self, boxes: List[Dict]) -> List[Dict]:
        """
        Extract headers using layout recognition results
        
        Args:
            boxes: List of text boxes from PDF parsing
            
        Returns:
            List of header items
        """
        headers = []
        
        # Find boxes marked as titles or with title-like characteristics
        for box in boxes:
            text = box["text"].strip()
            layout_type = box.get("layout_type", "")
            page_num = box.get("page_number", 1)
            
            # Skip if text is too short or appears to be noise
            if len(text) < 3 or self._is_noise_text(text):
                continue
            
            # Check for title layout type
            if layout_type == "title":
                headers.append({
                    "level": "H1",
                    "text": text,
                    "page": page_num
                })
            elif self._is_header_by_position_and_format(box, boxes):
                # Determine header level based on formatting/position
                level = self._determine_header_level(box, boxes)
                headers.append({
                    "level": level,
                    "text": text,
                    "page": page_num
                })
        
        # Sort by page and position
        headers.sort(key=lambda x: (x["page"], boxes[next(i for i, box in enumerate(boxes) 
                                                         if box["text"].strip() == x["text"])].get("top", 0)))
        
        return headers
    
    def _extract_headers_from_patterns(self, boxes: List[Dict]) -> List[Dict]:
        """
        Extract headers using text patterns and formatting cues
        
        Args:
            boxes: List of text boxes from PDF parsing
            
        Returns:
            List of header items
        """
        headers = []
        
        # Common header patterns
        header_patterns = [
            # Numbered sections
            (r'^\d+\.\s+(.+)$', 'H1'),
            (r'^\d+\.\d+\s+(.+)$', 'H2'), 
            (r'^\d+\.\d+\.\d+\s+(.+)$', 'H3'),
            
            # Lettered sections
            (r'^[A-Z]\.\s+(.+)$', 'H1'),
            (r'^[A-Z]\.\d+\s+(.+)$', 'H2'),
            
            # Roman numerals
            (r'^[IVX]+\.\s+(.+)$', 'H1'),
            
            # Common section names
            (r'^(introduction|background|methodology|methods|results|discussion|conclusion|references|appendix)$', 'H1'),
            (r'^(abstract|summary|overview)$', 'H1'),
        ]
        
        for box in boxes:
            text = box["text"].strip()
            page_num = box.get("page_number", 1)
            
            if len(text) < 3 or self._is_noise_text(text):
                continue
            
            # Check against patterns
            for pattern, level in header_patterns:
                match = re.match(pattern, text, re.IGNORECASE)
                if match:
                    header_text = match.group(1) if match.groups() else text
                    headers.append({
                        "level": level,
                        "text": header_text.strip(),
                        "page": page_num
                    })
                    break
            else:
                # Check if it looks like a header based on formatting
                if self._is_likely_header(text, box):
                    headers.append({
                        "level": "H2",  # Default level
                        "text": text,
                        "page": page_num
                    })
        
        # Remove duplicates and sort
        seen_texts = set()
        unique_headers = []
        for header in headers:
            if header["text"].lower() not in seen_texts:
                seen_texts.add(header["text"].lower())
                unique_headers.append(header)
        
        # Sort by page and position
        unique_headers.sort(key=lambda x: (x["page"], boxes[next(i for i, box in enumerate(boxes) 
                                                               if box["text"].strip() == x["text"])].get("top", 0)))
        
        return unique_headers
    
    def _is_noise_text(self, text: str) -> bool:
        """Check if text appears to be noise/irrelevant"""
        noise_patterns = [
            r'^\d+$',  # Just numbers
            r'^page\s+\d+',  # Page numbers
            r'^\d+\s*/\s*\d+$',  # Page x/y format
            r'^[^\w\s]*$',  # Only punctuation
            r'^\s*$',  # Only whitespace
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, text.lower()):
                return True
        return False
    
    def _is_header_by_position_and_format(self, box: Dict, all_boxes: List[Dict]) -> bool:
        """
        Determine if a box is likely a header based on position and formatting
        """
        text = box["text"].strip()
        
        # Check if it's at the beginning of a line/page
        same_page_boxes = [b for b in all_boxes if b.get("page_number") == box.get("page_number")]
        
        # Simple heuristic: if it's one of the first few text elements on a page
        # and doesn't look like body text, it might be a header
        if len(same_page_boxes) > 0:
            sorted_boxes = sorted(same_page_boxes, key=lambda x: x.get("top", 0))
            position = sorted_boxes.index(box) if box in sorted_boxes else -1
            
            if position <= 3 and self._is_likely_header(text, box):
                return True
        
        return False
    
    def _is_likely_header(self, text: str, box: Dict) -> bool:
        """
        Check if text/box is likely to be a header based on content
        """
        text = text.strip()
        
        # Length constraints
        if len(text) < 3 or len(text) > 150:
            return False
        
        # Check for header-like patterns
        header_indicators = [
            text.isupper() and len(text) > 5,  # ALL CAPS
            re.match(r'^\d+[\.\)]\s+\w+', text),  # Numbered sections
            re.match(r'^[A-Z][a-z].*[^\.!?]$', text),  # Sentence case, no ending punctuation
            any(keyword in text.lower() for keyword in 
                ['chapter', 'section', 'part', 'introduction', 'conclusion', 
                 'background', 'methodology', 'results', 'discussion'])
        ]
        
        return any(header_indicators)
    
    def _determine_header_level(self, box: Dict, all_boxes: List[Dict]) -> str:
        """
        Determine the hierarchical level of a header
        """
        text = box["text"].strip()
        
        # Pattern-based level detection
        if re.match(r'^\d+\.\s+', text):
            return "H1"
        elif re.match(r'^\d+\.\d+\s+', text):
            return "H2"
        elif re.match(r'^\d+\.\d+\.\d+\s+', text):
            return "H3"
        
        # Default to H2
        return "H2"


def main():
    """Test the header extraction pipeline"""
    pipeline = HeaderExtractionPipeline()
    
    # Test files
    test_files = [
        "file01.pdf",
        "file02.pdf", 
        "file03.pdf",
        "file04.pdf",
        "file05.pdf"
    ]
    
    for filename in test_files:
        try:
            result = pipeline.extract_document_structure(filename)
            print(f"\n=== {filename} ===")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    main()
