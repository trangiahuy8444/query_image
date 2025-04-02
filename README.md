# Visual Relationship Detection and Knowledge Graph Construction

This project implements a system for detecting visual relationships in images and constructing a knowledge graph using Neo4j. The system combines the RelTR (Transformer-based Visual Relationship Detection) model with graph database technology to create a comprehensive solution for understanding and querying visual relationships.

## Project Structure

```
neo4j/
├── RelTR/                     # Mã nguồn RelTR
├── app/                       # Ứng dụng web
│   ├── static/               # File tĩnh
│   ├── templates/            # Templates
│   └── __init__.py          # Khởi tạo Flask app
├── data/                     # Dữ liệu
│   ├── vg_1000/             # Dataset gốc
│   └── vg_1000_focused/     # Dataset đã xử lý
├── utils/                    # Các công cụ hỗ trợ
│   ├── data_processing.py
│   └── evaluation.py
├── README.md                 # Tài liệu
├── requirements.txt          # Dependencies
├── .gitignore               # Git ignore
└── app.py                   # Entry point
```

## Features

1. **Visual Relationship Detection**
   - Implementation of RelTR model for relationship detection
   - Custom dataset processing for focused relationship extraction
   - Evaluation metrics and visualization tools

2. **Knowledge Graph Construction**
   - Neo4j database integration
   - Automatic graph construction from detected relationships
   - CRUD operations for graph data management

3. **Web Interface**
   - Flask-based web application
   - Image upload and processing
   - Visual relationship visualization
   - Graph query interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd neo4j
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Neo4j:
   - Install Neo4j Desktop or use Neo4j Aura
   - Create a new database
   - Update connection settings in `app.py`

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Access the web interface at `http://localhost:5000`

3. Process new images:
   - Upload images through the web interface
   - View detected relationships
   - Query the knowledge graph

## Dataset

The project uses a focused subset of the Visual Genome dataset, processed to retain the most important relationships. The dataset processing pipeline includes:

1. Relationship importance scoring
2. Filtering based on scores
3. Maintaining data consistency
4. Optimizing for graph database storage

## Model Architecture

The system combines:
- RelTR: Transformer-based visual relationship detection
- Neo4j: Graph database for relationship storage and querying
- Flask: Web framework for user interface

## Evaluation

Evaluation metrics include:
- Relationship detection accuracy
- Graph construction efficiency
- Query response time
- System scalability

## Contributing

This project was developed as part of a thesis at Ho Chi Minh City University of Education. For any questions or suggestions, please contact the author.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- **Tran Quang Huy**
- Ho Chi Minh City University of Education
- Contact: [Your Email]

## Acknowledgments

- Thesis Advisor: [Advisor Name]
- Visual Genome Dataset Team
- Neo4j Community
- RelTR Paper Authors 