from setuptools import setup, find_packages

setup(
    name="resume-parser-kg",
    version="1.0.0",
    description="Resume Parser & Knowledge Graph Builder",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.28.1",
        "pypdf2==3.0.1",
        "python-docx==0.8.11",
        "openai==1.3.7",
        "anthropic==0.7.8",
        "google-generativeai==0.3.2",
        "neo4j==5.14.1",
        "python-dotenv==1.0.0",
        "pydantic==2.5.0",
        "typing-extensions==4.8.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
