This application allows users to upload PDF files, extract their content, and interact with the content using an AI-powered chatbot. It uses the OpenAI GPT-3.5 model to answer user questions about the uploaded PDFs, providing context-aware responses.
Upload PDFs: Upload one or more PDF files for analysis.
Extract Content: Extracts text content from the uploaded PDFs.
Ask Questions: Users can ask questions about the PDF content.
AI-Powered Responses: Uses OpenAI GPT-3.5 to provide intelligent, context-aware answers.
Chunk-Based Processing: Handles large documents by splitting them into manageable chunks for efficient embedding and querying.
Interactive UI: Built with Streamlit for an easy-to-use, interactive interface.
Dependencies: 
  streamlit, 
  PyPDF2, 
  pdfplumber, 
  sentence-transformers, 
  scikit-learn, 
  openai.
