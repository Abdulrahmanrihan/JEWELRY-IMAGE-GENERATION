# Jewelry Design Generator - Setup and Usage Guide

This guide explains how to set up and use the Jewelry Design Generator application on your Windows or Mac computer using the code from GitHub.

**Project Repository:** [Link to your GitHub repository here]

## Part 1: Setup (Do this only once)

**1. Install Prerequisite: Git**
   - If you don't have Git, download and install it from [https://git-scm.com/downloads](https://git-scm.com/downloads). Git is needed to copy the project files from GitHub.

**2. Install Python:**
   - If you don't have Python installed, download it from the official website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - **Recommended Version:** Python 3.9 or 3.10 (The app might work on others, but these are suggested for best compatibility).
   - **During Installation (Windows):** Make sure to check the box that says "Add Python to PATH".

**3. Get the Application Code from GitHub:**
   - Open Terminal (Mac) or Command Prompt/Git Bash (Windows).
   - Navigate to the directory where you want to store the project (e.g., `cd Documents` or `cd C:\Projects`).
   - Clone the repository using Git (Replace `<repo_url>` with the actual URL from GitHub):
     ```
     git clone <repo_url>
     ```
   - This will create a new folder (e.g., `JEWERY-GENERATION-main`). Navigate into this folder:
     ```
     cd <repository_folder_name>
     ```
     (e.g., `cd JEWERY-GENERATION-main`) You should now be inside the project folder containing `app.py`, `requirements.txt`, etc.

**4. Install PyTorch (AI Engine):**
   - Go to the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
   - Make the following selections on the page:
      - **PyTorch Build:** Stable
      - **Your OS:** Windows or Mac
      - **Package:** Pip
      - **Language:** Python
      - **Compute Platform:**
         - If you have a modern NVIDIA GPU and installed CUDA drivers: Select the latest CUDA version shown.
         - Otherwise (or if unsure): Select **CPU**.
   - Copy the command shown under "Run this Command". It will look something like `pip3 install torch torchvision torchaudio ...` (for CUDA) or `pip3 install torch torchvision torchaudio` (for CPU).
   - Paste the copied command into your Terminal/Command Prompt (while inside the project folder) and press Enter. Wait for installation to complete.

**5. Install Other Required Packages:**
   - In the same Terminal/Command Prompt window (still inside the project folder), run this command:
     ```
     pip install -r requirements.txt
     ```
   - This installs Streamlit, Diffusers, Transformers, and other necessary libraries, including `huggingface_hub` which is needed for the next step. Wait for all packages to download and install.

**6. Download AI Models:**
   - The AI models are required for generation but are too large for GitHub. You need to download them separately using the command line tool installed in the previous step.
   - Make sure you are still in the project's root folder in your Terminal/Command Prompt.
   - Run the following commands one by one:
     ```
     # Create the main 'models' directory
     mkdir models

     # Download the CLIP model (this might take some time)
     huggingface-cli download openai/clip-vit-base-patch32 --local-dir models/clip-vit-base-patch32 --local-dir-use-symlinks False

     # Download the Stable Diffusion model (this will take longer and requires more disk space)
     huggingface-cli download stabilityai/stable-diffusion-2-1 --local-dir models/stable-diffusion-2-1 --local-dir-use-symlinks False
     ```
   - Wait for both downloads to complete. They will save the models into the `models/clip-vit-base-patch32` and `models/stable-diffusion-2-1` subdirectories respectively. This download only needs to be done once.
   - **Disk Space:** Ensure you have sufficient disk space (around 10-15 GB might be needed for the models).

Setup is complete!

## Part 2: Running the Application

1.  **Open Terminal or Command Prompt** (if not already open).
2.  **Navigate to the Application Folder** (the one you cloned from GitHub, e.g., `cd C:\Projects\JEWERY-GENERATION-main`).
3.  **Run the App:** Type the following command and press Enter:
    ```
    streamlit run app.py
    ```
4.  Your web browser should automatically open with the Jewelry Design Generator interface. If not, the terminal will show a URL (like `http://localhost:8501`) - copy and paste this into your browser.

## Part 3: Using the Application

The application interface has two main areas: the **Sidebar** on the left for setup and the **Main Area** on the right for viewing images and results.

1.  **Create a Client Folder:**
    - In the sidebar under "Create New Client", type a name for your client (e.g., "Client ABC").
    - Click the "Create Client Folder" button.

2.  **Select a Client:**
    - In the sidebar under "Select Client", choose the desired client folder from the dropdown menu.

3.  **Upload Input Images:**
    - Once a client is selected, the sidebar will show "Upload Images for [Client Name]".
    - Click "Browse files" and select one or more jewelry images (.jpg, .jpeg, .png) from your computer. These are the images the AI will use for inspiration.
    - Uploaded images will appear in the "Input Images" tab in the main area.

4.  **Generate Designs:**
    - Click the "Generated Designs" tab in the main area.
    - **Choose Generation Method:**
        - `Standard`: Uses the *first* uploaded image as the main shape/structure, applying style from all images. Good for variations on one specific piece.
        - `Rotation`: Creates designs using *each* uploaded image as the base structure in turn. Good for exploring different base shapes.
        - `Batch Process`: Creates multiple designs *for each* uploaded image as the base. Generates the most designs.
    - **Adjust Parameters:**
        - `Style Transfer Strength`: Controls how much the AI changes the base image's style (higher value = more change). Start around 0.6-0.8.
        - `Number of Designs`: Set how many designs you want (total or per base image, depending on the method).
    - **Click the "✨ Generate Designs" Button.**

5.  **Wait for Processing:**
    - Generating designs takes time, especially if you don't have a powerful GPU (NVIDIA graphics card). It might take **several minutes per design** if running on CPU.
    - You'll see a spinner and progress messages. Please be patient and don't close the browser tab or terminal window.

6.  **View and Download Results:**
    - Once finished, the generated designs will appear under "Previously Generated Designs" in the "Generated Designs" tab.
    - Click the "Download All [Number] Designs as ZIP" button to save a zip file containing all the generated images for the selected client to your computer's Downloads folder.

**Troubleshooting:**

*   **Error `git: command not found`:** Git is not installed or not in your system's PATH. Install it from [https://git-scm.com/downloads](https://git-scm.com/downloads).
*   **Error `huggingface-cli: command not found`:** The `huggingface_hub` library didn't install correctly. Try running `pip install --force-reinstall huggingface_hub` and ensure step 5 completed without errors.
*   **Error "Model files not found":** The models were not downloaded correctly in Setup Step 6, or they are not in the expected `models/clip-vit-base-patch32` and `models/stable-diffusion-2-1` subdirectories. Double-check the folders exist and contain files. Re-run the `huggingface-cli download` commands if needed.
*   **Error "CUDA out of memory":** Your GPU doesn't have enough memory. Try generating fewer designs at once, or close other programs using the GPU. If you selected CUDA during PyTorch install but don't have a suitable GPU, you might need to reinstall PyTorch selecting the CPU option.
*   **Very Slow Generation:** This is expected if PyTorch was installed for CPU. Generating images is computationally intensive.
*   **Other Errors:** Note down any error messages shown in the app or the terminal window, as they can help diagnose the problem. Try restarting the application (close the terminal, then repeat Part 2). You can also report issues on the project's GitHub repository page.

