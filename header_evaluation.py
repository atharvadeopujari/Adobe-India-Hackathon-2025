import json
import argparse
from typing import Dict, List, Tuple, Any
from difflib import SequenceMatcher
import sys

class HeaderEvaluator:
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize the Header Evaluator
        
        Args:
            similarity_threshold: Minimum similarity score for text matching (0.0 to 1.0)
        """
        self.similarity_threshold = similarity_threshold
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison
        
        Args:
            text: Input text string
            
        Returns:
            Normalized text string
        """
        return text.lower().strip().replace('\n', ' ').replace('\t', ' ')
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        norm_text1 = self.normalize_text(text1)
        norm_text2 = self.normalize_text(text2)
        
        return SequenceMatcher(None, norm_text1, norm_text2).ratio()
    
    def load_json_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load JSON file and handle errors
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary containing JSON data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")
    
    def extract_headers_from_json(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract headers from various JSON formats
        
        Args:
            data: JSON data dictionary
            
        Returns:
            List of header dictionaries
        """
        headers = []
        
        # Check for 'outline' key (your format)
        if 'outline' in data:
            headers = data['outline']
        
        # Check for 'headers' key
        elif 'headers' in data:
            headers = data['headers']
        
        # Check for 'pages' key (layoutparser format)
        elif 'pages' in data:
            for page in data['pages']:
                if 'layout_data' in page and 'elements' in page['layout_data']:
                    elements = page['layout_data']['elements']
                    if 'titles' in elements:
                        for title in elements['titles']:
                            headers.append({
                                'level': 'H1',  # Default level for titles
                                'text': title.get('text', ''),
                                'page': page.get('page_number', 0)
                            })
        
        # Direct list format
        elif isinstance(data, list):
            headers = data
        
        # Ensure all headers have required fields
        normalized_headers = []
        for header in headers:
            if isinstance(header, dict):
                normalized_header = {
                    'level': header.get('level', 'H1'),
                    'text': header.get('text', ''),
                    'page': header.get('page', 0)
                }
                normalized_headers.append(normalized_header)
        
        return normalized_headers
    
    def find_best_match(self, target_header: Dict[str, Any], candidate_headers: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
        """
        Find the best matching header from candidates
        
        Args:
            target_header: Header to match
            candidate_headers: List of candidate headers
            
        Returns:
            Tuple of (best_match_header, similarity_score)
        """
        best_match = None
        best_score = 0.0
        
        for candidate in candidate_headers:
            # Calculate text similarity
            text_similarity = self.calculate_text_similarity(
                target_header['text'], 
                candidate['text']
            )
            
            # Bonus for exact level match
            level_bonus = 0.1 if target_header['level'] == candidate['level'] else 0.0
            
            # Bonus for same page
            page_bonus = 0.05 if target_header['page'] == candidate['page'] else 0.0
            
            # Total score
            total_score = text_similarity + level_bonus + page_bonus
            
            if total_score > best_score:
                best_score = total_score
                best_match = candidate
        
        return best_match, best_score
    
    def calculate_metrics(self, ground_truth: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate precision, recall, F1 score, and accuracy
        
        Args:
            ground_truth: List of ground truth headers
            predictions: List of predicted headers
            
        Returns:
            Dictionary containing metrics
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        matched_predictions = set()
        
        # For each ground truth header, find best match in predictions
        for gt_header in ground_truth:
            best_match, similarity = self.find_best_match(gt_header, predictions)
            
            if best_match and similarity >= self.similarity_threshold:
                true_positives += 1
                # Mark this prediction as matched
                pred_idx = next(i for i, p in enumerate(predictions) if p == best_match)
                matched_predictions.add(pred_idx)
            else:
                false_negatives += 1
        
        # Count unmatched predictions as false positives
        false_positives = len(predictions) - len(matched_predictions)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def evaluate(self, ground_truth_path: str, output_path: str) -> Dict[str, Any]:
        """
        Complete evaluation workflow
        
        Args:
            ground_truth_path: Path to ground truth JSON
            output_path: Path to output JSON
            
        Returns:
            Dictionary containing evaluation results
        """
        # Load data
        ground_truth_data = self.load_json_file(ground_truth_path)
        output_data = self.load_json_file(output_path)
        
        # Extract headers
        ground_truth_headers = self.extract_headers_from_json(ground_truth_data)
        predicted_headers = self.extract_headers_from_json(output_data)
        
        # Calculate metrics
        metrics = self.calculate_metrics(ground_truth_headers, predicted_headers)
        
        # Detailed analysis
        detailed_matches = []
        matched_predictions = set()
        
        for i, gt_header in enumerate(ground_truth_headers):
            best_match, similarity = self.find_best_match(gt_header, predicted_headers)
            
            match_info = {
                'ground_truth': gt_header,
                'best_match': best_match,
                'similarity': similarity,
                'is_match': similarity >= self.similarity_threshold
            }
            
            if best_match and similarity >= self.similarity_threshold:
                pred_idx = next(j for j, p in enumerate(predicted_headers) if p == best_match)
                matched_predictions.add(pred_idx)
            
            detailed_matches.append(match_info)
        
        # Find unmatched predictions
        unmatched_predictions = [
            predicted_headers[i] for i in range(len(predicted_headers)) 
            if i not in matched_predictions
        ]
        
        return {
            'metrics': metrics,
            'ground_truth_count': len(ground_truth_headers),
            'predicted_count': len(predicted_headers),
            'similarity_threshold': self.similarity_threshold,
            'detailed_matches': detailed_matches,
            'unmatched_predictions': unmatched_predictions
        }
    
    def print_evaluation_report(self, results: Dict[str, Any]) -> None:
        """
        Print a detailed evaluation report
        
        Args:
            results: Results from evaluate() method
        """
        metrics = results['metrics']
        
        print("=" * 50)
        print("HEADER EXTRACTION EVALUATION REPORT")
        print("=" * 50)
        
        print(f"\nDATA SUMMARY:")
        print(f"Ground Truth Headers: {results['ground_truth_count']}")
        print(f"Predicted Headers:    {results['predicted_count']}")
        print(f"Similarity Threshold: {results['similarity_threshold']}")
        
        print(f"\nMETRICS:")
        print("-" * 30)
        print(f"Precision:       {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall:          {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1 Score:        {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        print(f"\nCONFUSION MATRIX:")
        print("-" * 30)
        print(f"True Positives:  {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        
        # Detailed matches
        print(f"\nDETAILED MATCHES:")
        print("-" * 30)
        for i, match in enumerate(results['detailed_matches']):
            status = "✓ MATCH" if match['is_match'] else "✗ NO MATCH"
            print(f"{i+1}. {status} (similarity: {match['similarity']:.3f})")
            print(f"   GT: [{match['ground_truth']['level']}] \"{match['ground_truth']['text']}\" (page {match['ground_truth']['page']})")
            if match['best_match']:
                print(f"   PR: [{match['best_match']['level']}] \"{match['best_match']['text']}\" (page {match['best_match']['page']})")
            else:
                print(f"   PR: No match found")
            print()
        
        # Unmatched predictions
        if results['unmatched_predictions']:
            print(f"UNMATCHED PREDICTIONS:")
            print("-" * 30)
            for i, pred in enumerate(results['unmatched_predictions']):
                print(f"{i+1}. [{pred['level']}] \"{pred['text']}\" (page {pred['page']})")

def main():
    parser = argparse.ArgumentParser(description='Evaluate header extraction results')
    parser.add_argument('ground_truth', help='Path to ground truth JSON file')
    parser.add_argument('output', help='Path to output JSON file')
    parser.add_argument('--threshold', type=float, default=0.8, help='Similarity threshold (default: 0.8)')
    parser.add_argument('--save-results', help='Path to save detailed results JSON')
    parser.add_argument('--quiet', action='store_true', help='Only show metrics summary')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = HeaderEvaluator(similarity_threshold=args.threshold)
        
        # Run evaluation
        results = evaluator.evaluate(args.ground_truth, args.output)
        
        # Print report
        if not args.quiet:
            evaluator.print_evaluation_report(results)
        else:
            metrics = results['metrics']
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        # Save detailed results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed results saved to: {args.save_results}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()