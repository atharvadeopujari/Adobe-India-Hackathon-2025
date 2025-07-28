"""
Multilingual Support Module for Header Extraction
Provides language detection, text normalization, and multilingual pattern matching
while maintaining the structural analysis approach of the header extraction system.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class MultilingualTextProcessor:
    """
    Handles multilingual text processing for header extraction
    Focuses on structural patterns that work across languages
    """

    def __init__(self):
        """Initialize multilingual processor with language-specific patterns"""
        # Common header-indicating words across languages
        self.header_keywords = {
            'english': {
                'introduction', 'background', 'summary', 'conclusion', 'references',
                'acknowledgement', 'acknowledgment', 'overview', 'abstract', 'preface',
                'contents', 'chapter', 'section', 'appendix', 'bibliography', 'index',
                'methodology', 'results', 'discussion', 'findings', 'analysis'
            },
            'spanish': {
                'introducción', 'antecedentes', 'resumen', 'conclusión', 'referencias',
                'agradecimientos', 'resumen', 'abstracto', 'prefacio', 'contenidos',
                'capítulo', 'sección', 'apéndice', 'bibliografía', 'índice',
                'metodología', 'resultados', 'discusión', 'hallazgos', 'análisis'
            },
            'french': {
                'introduction', 'contexte', 'résumé', 'conclusion', 'références',
                'remerciements', 'aperçu', 'résumé', 'préface', 'sommaire',
                'chapitre', 'section', 'annexe', 'bibliographie', 'index',
                'méthodologie', 'résultats', 'discussion', 'conclusions', 'analyse'
            },
            'german': {
                'einführung', 'hintergrund', 'zusammenfassung', 'schlussfolgerung', 'referenzen',
                'danksagungen', 'überblick', 'abstrakt', 'vorwort', 'inhalt',
                'kapitel', 'abschnitt', 'anhang', 'bibliographie', 'index',
                'methodik', 'ergebnisse', 'diskussion', 'erkenntnisse', 'analyse'
            },
            'italian': {
                'introduzione', 'background', 'riassunto', 'conclusione', 'riferimenti',
                'ringraziamenti', 'panoramica', 'estratto', 'prefazione', 'contenuti',
                'capitolo', 'sezione', 'appendice', 'bibliografia', 'indice',
                'metodologia', 'risultati', 'discussione', 'risultati', 'analisi'
            },
            'portuguese': {
                'introdução', 'antecedentes', 'resumo', 'conclusão', 'referências',
                'agradecimentos', 'visão geral', 'resumo', 'prefácio', 'conteúdo',
                'capítulo', 'seção', 'apêndice', 'bibliografia', 'índice',
                'metodologia', 'resultados', 'discussão', 'descobertas', 'análise'
            },
            'dutch': {
                'inleiding', 'achtergrond', 'samenvatting', 'conclusie', 'referenties',
                'dankwoord', 'overzicht', 'abstract', 'voorwoord', 'inhoud',
                'hoofdstuk', 'sectie', 'bijlage', 'bibliografie', 'index',
                'methodologie', 'resultaten', 'discussie', 'bevindingen', 'analyse'
            },
            'russian': {
                'введение', 'предпосылки', 'резюме', 'заключение', 'ссылки',
                'благодарности', 'обзор', 'аннотация', 'предисловие', 'содержание',
                'глава', 'раздел', 'приложение', 'библиография', 'индекс',
                'методология', 'результаты', 'обсуждение', 'выводы', 'анализ'
            },
            'chinese': {
                '介绍', '背景', '摘要', '结论', '参考文献', '致谢', '概述', '摘要', '前言', '目录',
                '章节', '部分', '附录', '参考书目', '索引', '方法论', '结果', '讨论', '发现', '分析',
                '引言', '总结', '概要', '序言'
            },
            'japanese': {
                '紹介', '背景', '要約', '結論', '参考文献', '謝辞', '概要', '抄録', '序文', '目次',
                '章', 'セクション', '付録', '書誌', '索引', '方法論', '結果', '議論', '発見', '分析',
                'はじめに', 'まとめ', '概観', '序章'
            },
            'korean': {
                '소개', '배경', '요약', '결론', '참조', '감사의 말', '개요', '초록', '서문', '목차',
                '장', '섹션', '부록', '참고문헌', '색인', '방법론', '결과', '토론', '발견', '분석',
                '서론', '정리', '개관', '머리말'
            },
            'arabic': {
                'مقدمة', 'خلفية', 'ملخص', 'خاتمة', 'مراجع', 'شكر وتقدير', 'نظرة عامة', 'مستخلص', 'تمهيد', 'محتويات',
                'فصل', 'قسم', 'ملحق', 'ببليوغرافيا', 'فهرس', 'منهجية', 'نتائج', 'مناقشة', 'استنتاجات', 'تحليل'
            },
            'hindi': {
                'परिचय', 'पृष्ठभूमि', 'सारांश', 'निष्कर्ष', 'संदर्भ', 'आभार', 'अवलोकन', 'सार', 'प्रस्तावना', 'विषय-सूची',
                'अध्याय', 'खंड', 'परिशिष्ट', 'ग्रंथ-सूची', 'सूचकांक', 'कार्यप्रणाली', 'परिणाम', 'चर्चा', 'निष्कर्ष', 'विश्लेषण'
            }
        }

        # Compile all keywords for quick lookup
        self.all_header_keywords = set()
        for lang_keywords in self.header_keywords.values():
            self.all_header_keywords.update(lang_keywords)

        # Numbering patterns across different scripts
        self.numbering_patterns = [
            # Arabic numerals
            r'^\d+\.\s*',
            r'^\d+\.\d+\s*',
            r'^\d+\.\d+\.\d+\s*',
            r'^\d+\)\s*',
            r'^\(\d+\)\s*',

            # Roman numerals
            r'^[IVX]+\.\s*',
            r'^[ivx]+\.\s*',
            r'^[IVX]+\)\s*',
            r'^[ivx]+\)\s*',

            # Letters
            r'^[A-Z]\.\s*',
            r'^[a-z]\.\s*',
            r'^[A-Z]\)\s*',
            r'^[a-z]\)\s*',

            # Chinese numerals
            r'^[一二三四五六七八九十]+[、.]\s*',

            # Arabic-Indic numerals
            r'^[٠-٩]+[.]\s*',

            # Devanagari numerals
            r'^[०-९]+[.]\s*',
        ]

        # Script detection patterns
        self.script_patterns = {
            'latin': re.compile(r'[a-zA-ZÀ-ÿ]'),
            'cyrillic': re.compile(r'[а-яё]', re.IGNORECASE),
            'arabic': re.compile(r'[\u0600-\u06FF]'),
            'chinese': re.compile(r'[\u4e00-\u9fff]'),
            'japanese_hiragana': re.compile(r'[\u3040-\u309f]'),
            'japanese_katakana': re.compile(r'[\u30a0-\u30ff]'),
            'japanese_kanji': re.compile(r'[\u4e00-\u9faf]'),
            'korean': re.compile(r'[\uac00-\ud7af]'),
            'devanagari': re.compile(r'[\u0900-\u097f]'),
            'thai': re.compile(r'[\u0e00-\u0e7f]'),
            'hebrew': re.compile(r'[\u0590-\u05ff]'),
        }

    def detect_text_language(self, text: str) -> str:
        """
        Detect the primary script/language of the text
        Returns the detected script type
        """
        if not text or not text.strip():
            return 'unknown'

        text = text.strip()
        script_scores = {}

        # Count characters for each script
        for script, pattern in self.script_patterns.items():
            matches = pattern.findall(text)
            script_scores[script] = len(matches)

        if not script_scores or max(script_scores.values()) == 0:
            return 'unknown'

        # Return the script with the highest character count
        primary_script = max(script_scores, key=script_scores.get)

        # Special handling for Japanese (prioritize hiragana/katakana over kanji)
        if script_scores.get('japanese_hiragana', 0) > 0 or script_scores.get('japanese_katakana', 0) > 0:
            return 'japanese'

        return primary_script

    def normalize_text(self, text: str) -> str:
        """
        Normalize text for consistent processing across languages
        Handles different Unicode normalizations and whitespace
        """
        if not text:
            return ""

        # Unicode normalization (NFD: canonical decomposition)
        normalized = unicodedata.normalize('NFD', text)

        # Remove zero-width characters and normalize whitespace
        normalized = re.sub(r'[\u200b-\u200f\u2060\ufeff]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)

        return normalized.strip()

    def is_multilingual_header_keyword(self, text: str) -> bool:
        """
        Check if text contains header keywords in any supported language
        """
        normalized_text = self.normalize_text(text.lower())

        # Check for exact matches
        for keyword in self.all_header_keywords:
            if keyword in normalized_text:
                return True

        # Check for partial matches (for compound words)
        words = re.findall(r'\w+', normalized_text)
        for word in words:
            if word in self.all_header_keywords:
                return True

        return False

    def extract_multilingual_numbering(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extract numbering information from text in any supported numbering system
        Returns dict with numbering info if found, None otherwise
        """
        normalized_text = self.normalize_text(text)

        for pattern in self.numbering_patterns:
            match = re.match(pattern, normalized_text)
            if match:
                numbering = match.group(0).strip()
                remaining_text = normalized_text[len(numbering):].strip()

                return {
                    'numbering': numbering,
                    'text_without_numbering': remaining_text,
                    'pattern': pattern
                }

        return None

    def is_title_case_multilingual(self, text: str) -> bool:
        """
        Check if text follows title case patterns across different scripts
        """
        if not text:
            return False

        # Detect script
        script = self.detect_text_language(text)

        if script in ['latin', 'cyrillic']:
            # For Latin and Cyrillic scripts, check traditional title case
            words = text.split()
            if len(words) == 0:
                return False

            # Check if most words start with uppercase
            uppercase_words = sum(
                1 for word in words if word and word[0].isupper())
            return uppercase_words >= len(words) * 0.7

        elif script in ['chinese', 'japanese', 'korean']:
            # For CJK scripts, title case doesn't apply in the same way
            # Instead, check for consistent formatting and reasonable length
            return 5 <= len(text) <= 80

        elif script == 'arabic':
            # Arabic doesn't have case, so check other indicators
            return 5 <= len(text) <= 80 and not text.endswith('.')

        elif script == 'devanagari':
            # Devanagari doesn't have case, similar to Arabic
            return 5 <= len(text) <= 80

        return False

    def is_all_caps_multilingual(self, text: str) -> bool:
        """
        Check if text is in all caps (for scripts that have case)
        """
        if not text:
            return False

        script = self.detect_text_language(text)

        if script in ['latin', 'cyrillic']:
            # Check if all letters are uppercase
            letters = re.findall(r'[a-zA-ZÀ-ÿа-яё]', text, re.IGNORECASE)
            if not letters:
                return False
            uppercase_letters = [c for c in letters if c.isupper()]
            return len(uppercase_letters) >= len(letters) * 0.8

        # For scripts without case, return False
        return False

    def calculate_multilingual_header_score(self, text: str, font_size: float,
                                            is_bold: bool, dominant_font_size: float) -> float:
        """
        Calculate header likelihood score for multilingual text
        Uses the same structural approach as the original system
        """
        if not text:
            return 0.0

        score = 0.0
        normalized_text = self.normalize_text(text)

        # Font size factor (same as original)
        size_ratio = font_size / dominant_font_size
        if size_ratio >= 1.5:
            score += 0.5
        elif size_ratio >= 1.3:
            score += 0.4
        elif size_ratio >= 1.2:
            score += 0.3
        elif size_ratio >= 1.1:
            score += 0.2
        else:
            score -= 0.1

        # Bold bonus (same as original)
        if is_bold:
            score += 0.4

        # Multilingual numbering patterns
        numbering_info = self.extract_multilingual_numbering(normalized_text)
        if numbering_info:
            # Bonus based on numbering complexity
            if '.' in numbering_info['numbering']:
                dots = numbering_info['numbering'].count('.')
                if dots == 1:
                    score += 0.6  # "1. Title"
                elif dots == 2:
                    score += 0.5  # "1.1. Subtitle"
                elif dots == 3:
                    score += 0.4  # "1.1.1. Details"
            else:
                score += 0.3  # Other numbering patterns

        # Multilingual header keywords
        if self.is_multilingual_header_keyword(normalized_text):
            score += 0.3

        # Title case patterns (multilingual)
        if self.is_title_case_multilingual(normalized_text):
            score += 0.3

        # All caps (for applicable scripts)
        if self.is_all_caps_multilingual(normalized_text) and 5 <= len(normalized_text) <= 50:
            score += 0.4

        # Colon ending (common across languages)
        if normalized_text.endswith(':'):
            score += 0.3

        # Length penalties (same as original)
        if len(normalized_text) > 120:
            score -= 0.2
        if normalized_text.count(',') > 2:
            score -= 0.2

        # Penalty for common non-header patterns
        if re.search(r'\b(page|copyright|version|\d{4}|©)\b', normalized_text.lower()):
            score -= 0.3

        return max(0.0, min(score, 1.0))

    def get_language_info(self, text: str) -> Dict[str, str]:
        """
        Get comprehensive language information for a text
        """
        script = self.detect_text_language(text)
        normalized = self.normalize_text(text)
        numbering = self.extract_multilingual_numbering(text)

        return {
            'script': script,
            'normalized_text': normalized,
            'has_numbering': numbering is not None,
            'numbering_info': numbering,
            'has_header_keywords': self.is_multilingual_header_keyword(text),
            'is_title_case': self.is_title_case_multilingual(text),
            'is_all_caps': self.is_all_caps_multilingual(text)
        }
