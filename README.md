# Face Recognition App

## Prerequisites

Ensure you have Python 3.x installed. You can download it from [python.org](https://www.python.org/downloads/).

## Setting Up the Project

### 1. Create a Python Virtual Environment

#### On Windows:

```bash
python -m venv venv
```

#### On macOS/Linux:

```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment

#### On Windows:

```bash
venv\Scripts\activate
```

#### On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Required Packages

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Running the Project

After setting up the virtual environment and installing the dependencies, you can run the project:

```bash
python main.py
```

## Project Structure

The expected folder structure for images is as follows:

```
image_db/
    lec_num_1/
        student_id_1/
            image1.jpg
            image2.jpg
            ...
        student_id_2/
            ...
    lec_num_2/
        ...
```

Only `.jpeg`, `.jpg`, and `.png` image files are processed by default. You can customize the image types by modifying the `image_types` parameter in the `ImageEncoder` class.

## Logging

The application logs key events and errors during execution. Check the console for logging messages, which will help you troubleshoot any issues.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
