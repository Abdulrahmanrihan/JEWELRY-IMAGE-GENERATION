### **Project Overview**
We are creating a tool to automate the generation of jewelry designs using an AI image generation model. Each client will have their own unique set of input images, and the AI model will generate new designs that follow the style of those images. This will require an intuitive interface to manage the input-output process and allow the client to specify the number of generated images.

### **Key Features**
1. **Input Folder Structure:**
   - The `inputs` folder contains subfolders, each corresponding to a client. 
   - Each subfolder will hold the images of a specific client, which will be used to train the AI model and generate new images.
   - Users should be able to create new subfolders for each client inside the `inputs` folder.

2. **Output Folder Structure:**
   - The `outputs` folder will contain subfolders named after the client folders in `inputs`.
   - Inside each client-specific folder, the generated images will be saved.

3. **User Interface (UI):**
   - A simple UI where users can:
     - Select a client folder from the list of subfolders inside `inputs`.
     - Choose how many images to generate.
     - Trigger the AI model to generate the images.
   - After processing, a new folder with the same name as the selected client folder will be created in the `outputs` folder, containing the generated images.

### **Folder and File Structure**
```
/project-root
    /inputs
        /client_1
            image1.jpg
            image2.jpg
            ...
        /client_2
            image1.jpg
            image2.jpg
            ...
        ...
    /outputs
        /client_1
            output_image1.jpg
            output_image2.jpg
            ...
        /client_2
            output_image1.jpg
            output_image2.jpg
            ...
        ...
    main.py
    requirements.txt
    interface.py
```



### **Detailed Workflow**

1. **Creating a Client Folder:**
   - The user places client-specific images inside a new subfolder under `/inputs/`. The folder will have a meaningful name, such as `client_1`, `client_2`, etc.

2. **Interface Flow:**
   - When the user opens the interface, they should see a list of client subfolders under `inputs`.
   - The user selects the client folder they wish to work with.
   - The user specifies the number of output images to generate.
   - Upon clicking "Generate," the following steps should occur:
     - The tool reads the images in the selected folder.
     - The AI model processes these images to generate the desired number of new images based on the client's style.
     - A new folder with the same name as the client folder is created in the `outputs` directory.
     - The newly generated images are saved inside this folder.

3. **Post-Processing:**
   - Once the images are generated, the user should be able to open the `outputs` directory to view them.
   - If a client folder already exists in the `outputs` directory, it will be overwritten or a warning should be displayed.

