# 🏆 Adobe India Hackathon 2025 - Multilingual Document Header Extraction


## 🌟 Overview

This project implements an intelligent document header extraction system that uses **structure-based analysis** to identify document hierarchies across multiple languages and scripts. Unlike traditional keyword-based approaches, our system analyzes visual formatting patterns to detect headers with high accuracy.

### 🎯 Key Features

- **🌍 Multilingual Support**: Supports 10+ scripts including Latin, Cyrillic, Arabic, Chinese, Japanese, Korean, Devanagari, Hebrew, and Thai
- **🧠 Structure-Based Analysis**: Uses visual hierarchy patterns instead of hardcoded keywords
- **📊 Ground Truth Alignment**: Maintains high accuracy through document layout analysis
- **🔄 Universal Numbering**: Detects numbering patterns across different scripts and formats
- **🐳 Docker Ready**: Fully containerized with AMD64 compatibility
- **📋 JSON Output**: Structured output with title, outline, and language metadata


### Prerequisites

- Python 3.9+
- Docker (optional)
- PDF files for processing

### 🐳 Docker Usage (Recommended)

1. **Build and Run the Docker image:**
   ```bash
   docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
   docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
   ```


2. **Check results:**
   ```bash
   ls output/  # JSON files with extracted headers
   ```


## 🌍 Supported Languages & Scripts

| Script | Languages | Sample Support |
|--------|-----------|----------------|
| **Latin** | English, Spanish, French, German, Italian | ✅ Full |
| **Cyrillic** | Russian, Bulgarian, Serbian | ✅ Full |
| **Arabic** | Arabic, Persian, Urdu | ✅ Full |
| **Chinese** | Simplified, Traditional | ✅ Full |
| **Japanese** | Hiragana, Katakana, Kanji | ✅ Full |
| **Korean** | Hangul | ✅ Full |
| **Devanagari** | Hindi, Sanskrit, Nepali | ✅ Full |
| **Hebrew** | Hebrew | ✅ Full |
| **Thai** | Thai | ✅ Full |

## 📊 Output Format

```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Chapter 1: Introduction",
      "page": 1,
      "language": "latin"
    },
    {
      "level": "H2",
      "text": "1.1 Overview",
      "page": 1,
      "language": "latin"
    }
  ]
}
```

### Output Fields

- **`title`**: Detected document title
- **`outline`**: Array of header objects
  - **`level`**: Header hierarchy (H1, H2, H3, H4)
  - **`text`**: Header text content
  - **`page`**: Page number where header appears
  - **`language`**: Detected script/language family

## ⚙️ Configuration

### Environment Variables

```bash
# Logging level (DEBUG, INFO, WARNING, ERROR)
export LOG_LEVEL=INFO

# Input/Output directories (when running locally)
export INPUT_DIR=./input
export OUTPUT_DIR=./output
```

### Docker Environment

The Docker container is configured for AMD64 architecture:

```dockerfile
FROM --platform=linux/amd64 python:3.9-slim
```

## 🔧 Technical Details

### Core Components

1. **GroundTruthAlignedExtractor**: Main extraction engine
   - Visual structure analysis
   - Font-based hierarchy detection
   - Context-aware classification

2. **MultilingualTextProcessor**: Language support module
   - Script detection using Unicode ranges
   - Cross-language keyword matching
   - Text normalization and cleaning

3. **Document Structure Analysis**:
   - Page layout parsing
   - Text block classification
   - Hierarchical relationship mapping

### Performance Characteristics

- **Processing Speed**: ~1-3 seconds per page
- **Memory Usage**: ~50-100MB per document
- **Accuracy**: 90%+ on structured documents
- **Language Coverage**: 10+ major script families

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Adobe India Hackathon 2025

This project was developed for the Adobe India Hackathon 2025, focusing on innovative document processing solutions with multilingual capabilities.

**Team Repository**: [Adobe-India-Hackathon25](https://github.com/jhaaj08/Adobe-India-Hackathon25)

---

<div align="center">

**Made with ❤️ for Adobe India Hackathon 2025**

[🔗 Original Problem Statement](Problem%20Statement.pdf) | [📊 Implementation Details](IMPLEMENTATION_SUMMARY.md)

</div>
