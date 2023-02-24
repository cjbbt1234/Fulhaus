
# Furniture Recognition with ResNet18

This project is a simple image classification model built with PyTorch and ResNet18 to recognize furniture types such as chair, sofa, and bed.

## Prerequisites

-   Python 3.7 or higher
-   PyTorch
-   Flask
-   Torchvision
-   Pillow
-   Docker

## Installation

1.  Clone the repository:
    
    bashCopy code
    
    `git clone https://github.com/your-username/furniture-recognition.git` 
    
2.  Install the required packages

## Usage

You can also run the application in a Docker container. Follow these steps:

1.  Build the Docker image:
    
    `docker build -t furniture-recognition .` 
    
2.  Run the Docker container:
    
    `docker run -p 5000:5000 furniture-recognition` 
    
    This will start the container and map port 5000 in the container to port 5000 on the host machine.
    
3.  Follow steps 2-4 in the "Usage" section above to classify an image. The Docker container should be running at `http://localhost:5000/`.
    
## CI/CD Pipeline

This project includes a continuous integration and deployment (CI/CD) pipeline to automate the building, testing, and deployment process. The pipeline is configured using Github Actions and Docker.


## License

This project is licensed under the MIT License - see the `LICENSE` file for details.