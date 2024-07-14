# 🌟 QUILLERY - DeepMindCraft 🌟

Welcome to **QUILLERY by DeepMindCraft**, an advanced Retrieval-Augmented Generation (RAG) application built on Streamlit, leveraging the powerful capabilities of the Gemini API. This application combines the strengths of retrieval-based and generation-based AI models to provide precise and contextually relevant answers to your queries. 

## 🚀 Key Features

1. **Intuitive User Interface**: 
   - **Beautiful Design**: Incorporates neomorphic  design elements for an aesthetically pleasing user experience.
   - **Responsive Layout**: Ensures seamless interaction across different devices and screen sizes.

2. **Multi-format File Upload**:
   - Supports **CSV**, **PDF**, and **TXT** file formats, allowing you to analyze a variety of documents effortlessly.

3. **Advanced PDF Analysis**:
   - **Compare PDFs**: Analyze and compare multiple PDF documents side-by-side.
   - **Merge PDFs**: Combine multiple PDF documents and generate insights from the merged content.

4. **Generative AI Integration**:
   - Powered by the **Gemini API** for generating accurate and contextually relevant responses.
   - Utilizes **HuggingFace Embeddings** and **FAISS** for efficient document retrieval and similarity search.

5. **Interactive Query Handling**:
   - Enter your queries and receive detailed answers based on the content of your uploaded files.
   - Chat history feature to keep track of your interactions and responses.

## 🛠️ How It Works

1. **Upload Your Files**:
   - Navigate to the sidebar and select the file format (CSV, PDF, TXT).
   - Upload your files using the file uploader widget.

2. **Analyze and Query**:
   - For CSV files: View the dataframe head and tail, and query the content.
   - For PDF files: Choose between Compare or Merge analysis, and query the documents.
   - For TXT files: Perform text analysis and query the content.

3. **Generate Answers**:
   - The application uses the Gemini API to generate answers based on the content and your query.
   - View the generated responses and chat history within the application.

## 🎨 Design Elements

- **Neomorphic Design**: Soft UI elements that mimic physical objects for an intuitive experience.
- **Custom CSS**: Tailored CSS to enhance the visual appeal and usability of the application.

## 🔒 Security

- **Secure API Key Handling**: Your Gemini API key is securely managed and never exposed.

## 📜 Example Queries

- "Summarize the content of the PDF file."
- "Compare the main points between these two PDF documents."
- "Analyze the trends in this CSV file."
- "Provide insights based on the text document."

## 📈 Performance

- **Efficient Data Processing**: Optimized for handling large documents and complex queries.
- **Garbage Collection**: Ensures efficient memory management for a smooth user experience.

## 🚧 Memory Leakage Issues

Please note that **QUILLERY - DeepMindCraft** is currently deployed on the **free tier of the Streamlit Cloud**, which comes with certain memory limitations. Due to these constraints, the application may experience memory leakage issues, especially when handling large files or processing complex queries.

### What This Means:

- **Limited Session Duration**: Long sessions or multiple file uploads might lead to memory exhaustion, causing the application to restart.
- **Performance Impact**: Processing very large documents or numerous queries in a single session might degrade performance.
- **Data Loss**: Unexpected restarts due to memory limits can result in loss of unsaved data or chat history.

### Mitigation Measures:

- **Optimize File Size**: Try to upload smaller files to reduce memory usage.
- **Frequent Session Refresh**: Refresh the application periodically to clear memory and prevent leaks.
- **Efficient Query Handling**: Keep queries concise and to the point to minimize processing load.

We are continuously working on optimizing the application to mitigate these issues. For an enhanced experience, consider using a higher-tier deployment option with more generous memory limits.

---

Explore the power of **QUILLERY By DeepMindCraft** and elevate your document analysis with state-of-the-art AI technology. Whether you are a data analyst, researcher, or curious mind, this application is designed to provide you with the insights you need, quickly and accurately.

🔗 **Start Exploring Now!**
