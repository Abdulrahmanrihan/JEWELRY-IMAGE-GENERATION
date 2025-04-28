# --- Start of updated app.py ---
import os
import streamlit as st
from PIL import Image
import time
from glob import glob
import shutil
import sys
import re  # Import regular expressions for sorting product folders

# Set page config
st.set_page_config(
    page_title="Jewelry Design Generator",
    page_icon="💎",
    layout="wide"
)

# Try importing torch with error handling
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
except RuntimeError as e:
    st.error(f"Error initializing PyTorch: {str(e)}")
    device = "cpu"
    st.warning("Falling back to CPU mode due to PyTorch initialization error.")

# Import MultiImageStyleTransfer with error handling
try:
    from multiimage_style_transfer import MultiImageStyleTransfer
    model_import_success = True
except ImportError:
    st.error("Failed to import MultiImageStyleTransfer. Make sure 'multiimage_style_transfer.py' is in the same folder as app.py.")
    model_import_success = False
except Exception as e:
    st.error(f"An error occurred importing MultiImageStyleTransfer: {str(e)}")
    model_import_success = False

# Define project structure
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(ROOT_DIR, "inputs")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
MODELS_DIR = os.path.join(ROOT_DIR, "models")  # Define models directory path

# Create directories if they don't exist
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)  # Create models directory if it doesn't exist


# Function to get client folders
def get_client_folders():
    folders = [f for f in os.listdir(INPUTS_DIR)
               if os.path.isdir(os.path.join(INPUTS_DIR, f))]
    return sorted(folders)


# Function to create a new client folder
def create_client_folder(folder_name):
    # Sanitize folder name (basic example, can be enhanced)
    safe_folder_name = "".join(c for c in folder_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if not safe_folder_name:
        st.error("Invalid folder name.")
        return False

    new_folder_path = os.path.join(INPUTS_DIR, safe_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        st.success(f"Created client folder: {safe_folder_name}")
        return True
    else:
        st.error(f"Folder '{safe_folder_name}' already exists!")
        return False


# Function to get product folders for a client
def get_product_folders(client_folder):
    client_output_path = os.path.join(OUTPUTS_DIR, client_folder)
    if not os.path.exists(client_output_path):
        return []  # Return an empty list if the client folder doesn't exist yet

    folders = [f for f in os.listdir(client_output_path)
               if os.path.isdir(os.path.join(client_output_path, f))]
    return sorted(folders)


# Function to create a new product folder for a client
def create_product_folder(client_folder, product_name):
    # Sanitize the product name (similar to client folders)
    safe_product_name = "".join(c for c in product_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if not safe_product_name:
        st.error("Invalid product name.")
        return False

    client_output_path = os.path.join(OUTPUTS_DIR, client_folder)
    new_folder_path = os.path.join(client_output_path, safe_product_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path, exist_ok=True) # create directory (if doesn't exist)
        st.success(f"Created product folder: {safe_product_name} for client {client_folder}")
        return True
    else:
        st.error(f"Product folder '{safe_product_name}' already exists for client {client_folder}!")
        return False


# Function to count images in a folder
def count_images_in_folder(folder_path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(folder_path, ext)))
    return len(image_files)


# Function to display images in a folder
def display_folder_images(folder_path, columns=3):
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(folder_path, ext)))

    if not image_files:
        st.info("No images found in this folder.")
        return

    # Display images in a grid
    cols = st.columns(columns)
    for i, image_path in enumerate(image_files):
        try:
            with cols[i % columns]:
                img = Image.open(image_path)
                st.image(img, caption=os.path.basename(image_path), use_column_width=True)
        except Exception as e:
            st.error(f"Could not load image: {os.path.basename(image_path)}. Error: {e}")


# Function to upload images to a client folder
def upload_images_to_folder(folder_path):
    uploaded_files = st.file_uploader("Upload images for the client",
                                      accept_multiple_files=True,
                                      type=["jpg", "jpeg", "png"],
                                      key=f"uploader_{os.path.basename(folder_path)}")  # Unique key per folder

    if uploaded_files:
        saved_count = 0
        for uploaded_file in uploaded_files:
            try:
                # Sanitize filename
                safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in ('.', '_', '-')).rstrip()
                file_path = os.path.join(folder_path, safe_filename)

                # Avoid overwriting existing files unintentionally
                counter = 1
                base, ext = os.path.splitext(file_path)
                while os.path.exists(file_path):
                    file_path = f"{base}_{counter}{ext}"
                    counter += 1

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_count += 1
            except Exception as e:
                st.error(f"Failed to save {uploaded_file.name}. Error: {e}")

        if saved_count > 0:
            st.success(f"Uploaded {saved_count} images!")
            st.rerun()  # Refresh page to show new images


# *** MODIFIED FUNCTION ***
def generate_designs(client_folder, product_folder, generation_method, params):
    if not model_import_success:
        st.error("Model support library could not be imported. Please check the setup.")
        return False

    input_folder = os.path.join(INPUTS_DIR, client_folder)
    # *** MODIFIED: Output path is now to the selected product folder ***
    output_folder = os.path.join(OUTPUTS_DIR, client_folder, product_folder)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get input images
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    input_images_paths = []
    for ext in image_extensions:
        input_images_paths.extend(glob(os.path.join(input_folder, ext)))

    if not input_images_paths:
        st.error("No input images found in the selected client's folder!")
        return False

    # Initialize the style transfer model (using local path)
    try:
        model = MultiImageStyleTransfer(model_dir=MODELS_DIR)
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error(f"Please ensure the 'models' folder exists and contains 'clip-vit-base-patch32' and 'stable-diffusion-2-1' subfolders with model files.")
        return False
    except RuntimeError as e:
        if "no running event loop" in str(e) and sys.version_info >= (3, 12):
            st.error("PyTorch event loop error detected, potentially due to Python 3.12. Consider using Python 3.9 or 3.10.")
            return False
        elif "CUDA out of memory" in str(e):
            st.error("GPU ran out of memory. Try reducing image/design count, or close other GPU apps.")
            return False
        else:
            st.error(f"A PyTorch runtime error occurred: {str(e)}")
            st.exception(e)
            return False
    except Exception as e:
        st.error(f"An unexpected error occurred initializing the model: {str(e)}")
        st.exception(e)
        return False

    # --- Generation logic (largely unchanged) ---
    try:
        with st.spinner(f"Generating designs into '{product_folder}'... This may take several minutes."):
            variations = []
            start_time = time.time()

            if generation_method == "Standard (Single Base)":
                st.info(f"Generating {params['num_variations']} variations using '{os.path.basename(input_images_paths[0])}' as base...")
                variations = model.generate_multi_style_variations(
                    input_images_paths,
                    num_variations=params["num_variations"],
                    strength=params["strength"],
                    use_rotation=False
                )
            elif generation_method == "Rotation (Multiple Base)":
                st.info(f"Generating {params['num_variations']} variations, rotating through base images...")
                variations = model.generate_multi_style_variations(
                    input_images_paths,
                    num_variations=params["num_variations"],
                    strength=params["strength"],
                    use_rotation=True
                )
            elif generation_method == "Batch Process (All Bases)":
                st.info(f"Generating {params['variations_per_base']} variations for each of the {len(input_images_paths)} base images...")
                all_variations = []
                num_bases = len(input_images_paths)
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i in range(num_bases):
                    current_base_path = input_images_paths[i]
                    status_text.text(f"Processing base image {i+1}/{num_bases}: {os.path.basename(current_base_path)}")
                    ordered_paths = [current_base_path] + [p for j, p in enumerate(input_images_paths) if j != i]
                    base_variations = model.generate_multi_style_variations(
                        ordered_paths,
                        num_variations=params["variations_per_base"],
                        strength=params["strength"],
                        use_rotation=False
                    )
                    all_variations.extend(base_variations)
                    progress_bar.progress((i + 1) / num_bases)

                variations = all_variations
                status_text.text("Batch processing complete.")

            # --- MODIFIED: Save variations to the selected output folder ---
            saved_count = 0
            for i, variation in enumerate(variations):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_filename = f"design_{timestamp}_{i + 1}.png"
                # Save to the selected output_folder
                output_path = os.path.join(output_folder, output_filename)
                try:
                    variation.save(output_path)
                    saved_count += 1
                except Exception as e:
                    st.error(f"Failed to save generated image {output_filename}. Error: {e}")

            end_time = time.time()
            duration = end_time - start_time

        if saved_count > 0:
            st.success(
                f"Successfully generated and saved {saved_count} designs into '{product_folder}' in {duration:.2f} seconds!")
            return True
        else:
            st.error("No designs were generated or saved successfully.")
            return False

    except Exception as e:
        st.error(f"An error occurred during design generation: {str(e)}")
        st.error("Details: Check GPU memory (if applicable), input image formats, and file permissions.")
        st.exception(e)
        return False


# Main app layout function
def main():
    st.title("💎 Jewelry Design Generator")

    if sys.version_info >= (3, 12):
        st.warning(
            "Note: Python 3.12 detected. Consider Python 3.9 or 3.10 for potentially better compatibility with some AI libraries.")
    elif sys.version_info < (3, 8):
        st.warning("Note: Python version older than 3.8 detected. Consider updating Python.")

    # Sidebar for controls
    st.sidebar.header("Client Management")

    # Create new client folder
    st.sidebar.subheader("Create New Client")
    new_client_name = st.sidebar.text_input("Enter New Client Name")
    if st.sidebar.button("Create Client Folder"):
        if new_client_name:
            if create_client_folder(new_client_name):
                st.rerun()
        else:
            st.sidebar.error("Please enter a client name.")

    # Client selection
    st.sidebar.subheader("Select Client")
    client_folders = get_client_folders()

    if not client_folders:
        st.info("No client folders found. Create a client folder using the sidebar to begin.")
        return

    selected_client_index = st.sidebar.selectbox(
        "Client",
        range(len(client_folders)),
        format_func=lambda i: client_folders[i],
        index=0
    )
    selected_client = client_folders[selected_client_index]

    # --- NEW: Product Folder Management ---
    st.sidebar.subheader(f"Product Management for {selected_client}")
    # Create new product folder
    new_product_name = st.sidebar.text_input("Enter New Product Name")
    if st.sidebar.button("Create Product Folder"):
        if new_product_name:
            if create_product_folder(selected_client, new_product_name):
                st.rerun()  # Refresh to update product folder list
        else:
            st.sidebar.error("Please enter a product name.")

    # Select existing product folder
    product_folders = get_product_folders(selected_client)
    if not product_folders:
        st.info(f"No product folders found for {selected_client}. Create one to begin.")
        selected_product = None # no product folder to select, set to None
    else:
        selected_product = st.sidebar.selectbox(
            "Select Product Folder",
            product_folders,
            index=0 # default selection
        )
    # --- END NEW ---

    client_folder_path = os.path.join(INPUTS_DIR, selected_client)

    st.sidebar.subheader(f"Upload Images for {selected_client}")
    upload_images_to_folder(client_folder_path)

    # Main area display
    st.header(f"Workspace for Client: {selected_client}")
    tab1, tab2 = st.tabs(["Input Images", "Generated Designs"])

    with tab1:
        st.subheader("Input Images")
        num_images = count_images_in_folder(client_folder_path)
        st.write(f"Number of uploaded images: {num_images}")
        if num_images > 0:
            display_folder_images(client_folder_path)
        else:
            st.info(f"No images uploaded yet for '{selected_client}'. Use the sidebar to upload images.")

    with tab2:
        st.subheader("Generate New Designs")
        num_input_images = count_images_in_folder(client_folder_path)
        if num_input_images == 0:
            st.warning("Please upload at least one input image for the selected client before generating designs.")
        elif selected_product is None:
            st.warning("Please create or select a product folder before generating designs.") # requires a product folder to proceed
        else:
            generation_method = st.selectbox(
                "Generation Method",
                ["Standard (Single Base)", "Rotation (Multiple Base)", "Batch Process (All Bases)"],
                index=0,
                help="Determines how uploaded images are used as base structure and style references."
            )
            # Method descriptions... (kept concise)
            if generation_method == "Standard (Single Base)": st.caption(
                "Uses the *first* image as structure, style from *all*.")
            elif generation_method == "Rotation (Multiple Base)": st.caption(
                "Rotates base structure through *each* image, style from *all*.")
            elif generation_method == "Batch Process (All Bases)": st.caption(
                "Generates designs *for each* image as base structure, style from *all*.")

            st.subheader("Generation Parameters")
            col1, col2 = st.columns(2)
            with col1:
                strength = st.slider("Style Transfer Strength", 0.1, 0.95, 0.7, 0.05)
            params = {"strength": strength}
            if generation_method in ["Standard (Single Base)", "Rotation (Multiple Base)"]:
                with col2:
                    num_variations = st.number_input("Total Number of Designs", 1, 20, 3, 1)
                params["num_variations"] = num_variations
            elif generation_method == "Batch Process (All Bases)":
                with col2:
                    variations_per_base = st.number_input("Designs Per Base Image", 1, 10, 2, 1)
                params["variations_per_base"] = variations_per_base
                total_designs_batch = variations_per_base * num_input_images
                st.caption(f"This will generate a total of {total_designs_batch} designs.")

            if st.button("✨ Generate Designs", type="primary", key="generate_button"):
                if num_input_images > 0 and selected_product: # ensure a product folder is selected before proceeding
                    if device == 'cpu':
                        st.warning("⚠️ Running on CPU: Generation will be very slow.")
                    elif num_input_images > 5 or params.get('num_variations', 0) > 5 or params.get(
                            'variations_per_base', 0) * num_input_images > 10:
                        st.warning(f"⏳ Processing multiple images/designs may take significant time. Please be patient.")
                    success = generate_designs(selected_client, selected_product, generation_method, params)
                    if success:
                        st.balloons()
                        st.rerun()
                else:
                    st.error("Please upload images AND select a product folder before generating designs!")

        # --- MODIFIED: Display generated designs section ---
        st.divider()
        st.subheader("Previously Generated Designs")
        # Find base path to selected Client's output folder
        client_output_base_path = os.path.join(OUTPUTS_DIR, selected_client)

        # Check if any output folders exists for client
        if os.path.exists(client_output_base_path):
            # Product folder list, includes handling for possible errors when accessing system files
            product_dirs = []
            try:
                product_dirs = sorted([
                    d for d in os.listdir(client_output_base_path)
                    if os.path.isdir(os.path.join(client_output_base_path, d))
                ])
            except Exception as e:
                 st.warning(f"Error when listing or sorting product directories: {e}")
            # Handle case where no output folders exist
            if product_dirs:
                total_designs_across_products = 0
                # Go through each product folder
                for product_dir_name in product_dirs:
                    product_dir_path = os.path.join(client_output_base_path, product_dir_name)
                    num_images_in_product = count_images_in_folder(product_dir_path)
                    # Display generated images if found, also updates design count
                    if num_images_in_product > 0:
                        total_designs_across_products += num_images_in_product
                        st.markdown(f"#### Product: {product_dir_name} ({num_images_in_product} images)")
                        display_folder_images(product_dir_path, columns=4)
                        st.divider()
            # Check if designs were added
            if total_designs_across_products > 0:
                st.write(f"Total number of generated designs across all products: {total_designs_across_products}")
            else:
                st.info("No designs generated for this client yet.")
        else:
            st.info("No designs generated for this client yet.") # when client output path doesn't exist

        # Add a download button for all designs - download a ZIP folder containing the sub folders
        if os.path.exists(client_output_base_path):
            zip_filename = f"{selected_client}_all_designs.zip"
            # Define where the temporary zip file will be created (outside outputs)
            zip_path_base = os.path.join(ROOT_DIR, f"{selected_client}_all_designs_temp")

            try:
                # Ensure the zip doesn't exist before creating it
                if os.path.exists(zip_path_base + ".zip"):
                    os.remove(zip_path_base + ".zip")

                # Create the zip archive containing the client's entire output folder
                # The base_dir is where the archive is created, root_dir is what to zip
                shutil.make_archive(
                    base_name=zip_path_base,  # Path and name of the archive to create (without .zip)
                    format='zip',  # Archive format
                    root_dir=client_output_base_path  # The directory whose contents will be zipped
                    # base_dir=None # Not needed if root_dir is set correctly relative to files
                )
                zip_path_full = zip_path_base + ".zip"  # Full path to the created zip

                if os.path.exists(zip_path_full):
                    with open(zip_path_full, "rb") as f:
                        st.download_button(
                            label=f"Download All {total_designs_across_products} Designs (incl. all products) as ZIP",
                            data=f,
                            file_name=zip_filename,  # The name the user sees when downloading
                            mime="application/zip",
                            key="download_zip_all"
                        )
                else:
                    st.error("Failed to create the ZIP file for download.")

            except Exception as e:
                st.error(f"Could not create ZIP file for download. Error: {e}")
                st.exception(e)  # Show details for debugging

    # Sidebar Footer Info
    st.sidebar.divider()
    st.sidebar.header("How To Use")
    st.sidebar.info("""
    1.  **Create/Select Client** using the sidebar.
    2.  **(NEW) Create/Select Product Folder:** Create product folders to organize designs.
    3.  **Upload Images** for the selected client.
    4.  **Go to 'Generated Designs' Tab:** Configure method & parameters.
    5.  **Click 'Generate Designs'.** Wait for processing.
    6.  **View Results:** Designs appear below, organized by product folder.
    7.  **Download:** Use the 'Download All' button for a ZIP file.
    """)
    st.sidebar.divider()
    st.sidebar.caption(f"Using Device: {device.upper()} | Python: {sys.version.split()[0]}")
    st.sidebar.caption(f"Models loaded from: ./models/")


# Entry point
if __name__ == "__main__":
    main()
# --- End of updated app.py ---
