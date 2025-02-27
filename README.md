# Jacie - Your Financial Assistant 💼

Jacie is a financial assistant application designed to help users manage and analyze financial data efficiently. Built with Streamlit, it provides a user-friendly interface for interacting with various financial tools and services.

## Table of Contents 📑
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Known Issues](#known-issues)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Features ✨

- **Financial Analysis** 📊: Utilize advanced algorithms to analyze financial data and provide insights. For example, Jacie can generate reports on spending patterns and investment performance.
- **Image Processing** 🖼️: Process and analyze PDF images to extract relevant financial information. This feature supports various formats and provides accurate data extraction.
- **Conversational Interface** 💬: Engage with a chatbot powered by Google Vertex AI for financial queries. The chatbot can answer questions about financial terms, provide market updates, and more.
- **Memory-Enhanced Queries** 🧠: Enhance user queries using conversation history for more accurate responses. This feature allows for more context-aware interactions.

## Installation 🛠️

To run Jacie, ensure you have Python installed and follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd jacie
   ```

2. Install Poetry if not already installed:
   ```bash
   pip install poetry
   ```

3. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

4. Set up Google Cloud credentials:
   - Ensure your Google Cloud credentials are available in `credentials.json` or set in Streamlit secrets.

## Usage 🚀

Run the application using Streamlit:

```bash
streamlit run jacie/app.py
```

### Example Usage
1. Open the application in your browser.
2. Navigate to the financial analysis section to upload your data.
3. Use the chatbot to ask questions about your financial data.

## Configuration ⚙️

- **Environment Variables**: Set `GOOGLE_APPLICATION_CREDENTIALS` in your environment or Streamlit secrets for Google Cloud access.

## Testing 🧪

To run tests, use the following command:

```bash
pytest tests/
```

## Known Issues 🐞

- Some PDF formats may not be fully supported. Please report any issues on the GitHub repository.

## Contributing 🤝

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact 📧

For any inquiries or support, please contact [elroy.galbraith@aeontsolutions.com].

## Acknowledgments 🙏

- Thanks to the developers of Streamlit and Langchain for their excellent tools.
