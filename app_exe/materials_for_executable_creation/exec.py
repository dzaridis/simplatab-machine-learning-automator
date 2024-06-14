import os
import subprocess
import webbrowser
import time
from tkinter import Tk, Label, Button, filedialog, Frame, messagebox

class DockerApp:
    def __init__(self, master):
        self.master = master
        master.title("Simplatab Machine Learning Automator")

        # Line for citation
        self.citation_label = Label(master, text="Please install Docker Desktop to run the tool. If you proceed to run it without docker desktop you will be prompted on the page to download it.", bg='#ffffff')
        self.citation_label.pack(side='top', pady=10)

        # Create a frame to hold the widgets
        self.frame = Frame(master, bg='#ffffff', bd=5)
        self.frame.place(relx=0.5, rely=0.5, anchor='center')

        # Add a title label
        self.title_label = Label(self.frame, text="Select the input and output folders for the Docker container.", bg='#ffffff')
        self.title_label.pack(pady=10)

        # Add explanatory text for the input folder
        self.input_explanation = Label(self.frame, text="Select the folder containing the Train.csv and Test.csv files.", bg='#ffffff')
        self.input_explanation.pack(pady=5)

        # Button to select input folder
        self.input_button = Button(self.frame, text="Select Input Folder", command=self.select_input_folder)
        self.input_button.pack(pady=5)

        # Add explanatory text for the output folder
        self.output_explanation = Label(self.frame, text="Select the folder where the output files will be saved.", bg='#ffffff')
        self.output_explanation.pack(pady=5)

        # Button to select output folder
        self.output_button = Button(self.frame, text="Select Output Folder", command=self.select_output_folder)
        self.output_button.pack(pady=5)

        # Button to run Docker Compose
        self.run_button = Button(self.frame, text="Run the Tool", command=self.run_docker_compose)
        self.run_button.pack(pady=20)

        # Line for citation
        self.citation_label = Label(master, text="If you use this tool, please cite our work.", bg='#ffffff')
        self.citation_label.pack(side='bottom', pady=10)

        self.input_folder = ""
        self.output_folder = ""

    def select_input_folder(self):
        self.input_folder = filedialog.askdirectory()
        print(f"Selected input folder: {self.input_folder}")

    def select_output_folder(self):
        self.output_folder = filedialog.askdirectory()
        print(f"Selected output folder: {self.output_folder}")

    def run_docker_compose(self):
        if not self.input_folder or not self.output_folder:
            print("Please select both input and output folders.")
            return

        # Check if Docker Desktop is installed
        if not self.is_docker_installed():
            self.prompt_docker_installation()
            return

        # Start Docker Desktop if it's not running
        self.start_docker_desktop()

        # Save the selected folders to environment variables
        os.environ['INPUT_FOLDER'] = self.input_folder
        os.environ['OUTPUT_FOLDER'] = self.output_folder

        # Write the folder paths to a .env file
        with open('.env', 'w') as env_file:
            env_file.write(f"INPUT_FOLDER={self.input_folder}\n")
            env_file.write(f"OUTPUT_FOLDER={self.output_folder}\n")

        # Check if Docker image exists locally
        image_name = "dimzaridis/simplatab-machine-learning-automator:0.7.4-TestVersion"
        image_exists = self.check_docker_image_exists(image_name)

        if not image_exists:
            # Pull the Docker image if it doesn't exist locally
            docker_pull_command = f"docker pull {image_name}"
            print(f"Pulling Docker image: {docker_pull_command}")
            subprocess.run(docker_pull_command, shell=True)

        # Run Docker Compose
        docker_compose_file = os.path.join(os.path.dirname(__file__), 'docker-compose.yml')
        docker_compose_command = f"docker-compose -f {docker_compose_file} up -d"
        print(f"Running command: {docker_compose_command}")
        subprocess.run(docker_compose_command, shell=True)

        # Inform the user about redirection
        print("Docker is starting. You will be redirected to the web application in 2-3 minutes.")
        self.run_button.config(text="Docker is starting... Please wait.")

        # Open the browser to localhost:8000 after a delay
        self.master.after(10000, self.open_browser)  # 180000 milliseconds = 3 minutes

    def is_docker_installed(self):
        try:
            subprocess.run("docker --version", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False

    def prompt_docker_installation(self):
        messagebox.showinfo("Docker Not Found", "Docker Desktop is not installed. You will be redirected to the Docker Desktop download page.")
        webbrowser.open("https://www.docker.com/products/docker-desktop")

    def start_docker_desktop(self):
        try:
            # Check if Docker is running by executing a simple Docker command
            subprocess.run("docker info", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Docker is already running.")
        except subprocess.CalledProcessError:
            print("Docker is not running. Starting Docker Desktop...")
            # Start Docker Desktop
            docker_desktop_path = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"  # Update the path if needed
            subprocess.Popen([docker_desktop_path])
            # Wait for Docker to start
            time.sleep(40)  # Adjust the sleep time as needed to ensure Docker starts

    def check_docker_image_exists(self, image_name):
        result = subprocess.run(f"docker images -q {image_name}", shell=True, capture_output=True, text=True)
        return result.stdout.strip() != ""

    def open_browser(self):
        webbrowser.open("http://localhost:5000")
        self.run_button.config(text="Run the Tool")

if __name__ == "__main__":
    root = Tk()
    root.geometry("800x600")  # Set the window size
    docker_app = DockerApp(root)
    root.mainloop()