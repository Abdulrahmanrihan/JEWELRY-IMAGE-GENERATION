import os
import streamlit as st
from PIL import Image
import time
from glob import glob
import shutil
import sys
import re

# Set page config
st.set_page_config(
    page_title="디자인 생성 인터페이스",
    page_icon="",
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

# Import MultiImageStyleTransfer with improved error handling
try:
    # Add the current directory to the path to ensure module can be found
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from multiimage_style_transfer import MultiImageStyleTransfer
    model_import_success = True
except ImportError:
    try:
        # Try relative import as fallback
        from .multiimage_style_transfer import MultiImageStyleTransfer
        model_import_success = True
    except ImportError:
        st.error("Failed to import MultiImageStyleTransfer. Make sure 'multiimage_style_transfer.py' is in the same folder as app.py.")
        model_import_success = False
except Exception as e:
    st.error(f"Error occurred importing MultiImageStyleTransfer: {str(e)}")
    model_import_success = False

# Define project structure
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(ROOT_DIR, "inputs")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
GENERATED_IMAGES_DIR = os.path.join(ROOT_DIR, "generated_images")

# Create directories if they don't exist
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)

# Function to get client folders
def get_client_folders():
    folders = [f for f in os.listdir(INPUTS_DIR)
               if os.path.isdir(os.path.join(INPUTS_DIR, f))]
    return sorted(folders)

# Function to create a new client folder
def create_client_folder(folder_name):
    safe_folder_name = "".join(c for c in folder_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if not safe_folder_name:
        st.error("Invalid folder name.")
        return False

    new_folder_path = os.path.join(INPUTS_DIR, safe_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        # Also create client folder in generated_images directory
        os.makedirs(os.path.join(GENERATED_IMAGES_DIR, safe_folder_name), exist_ok=True)
        st.success(f"Created client folder: {safe_folder_name}")
        return True
    else:
        st.error(f"Folder '{safe_folder_name}' already exists!")
        return False

# Function to get product folders for a client in input or output directory
def get_product_folders(client_folder, base_dir):
    client_path = os.path.join(base_dir, client_folder)
    if not os.path.exists(client_path):
        return []
    folders = [f for f in os.listdir(client_path)
               if os.path.isdir(os.path.join(client_path, f))]
    return sorted(folders)

# Function to create new product folders for a client in both input and output directories
def create_product_folder(client_folder, product_name):
    safe_product_name = "".join(c for c in product_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if not safe_product_name:
        st.error("Invalid product name.")
        return False
    
    input_product_path = os.path.join(INPUTS_DIR, client_folder, safe_product_name)
    output_product_path = os.path.join(OUTPUTS_DIR, client_folder, safe_product_name)
    generated_product_path = os.path.join(GENERATED_IMAGES_DIR, client_folder, safe_product_name)
    
    success = True
    if not os.path.exists(input_product_path):
        try:
            os.makedirs(input_product_path, exist_ok=True)
        except Exception as e:
            st.error(f"Failed to create input product folder: {e}")
            success = False

    if not os.path.exists(output_product_path):
        try:
            os.makedirs(output_product_path, exist_ok=True)
        except Exception as e:
            st.error(f"Failed to create output product folder: {e}")
            success = False
            
    if not os.path.exists(generated_product_path):
        try:
            os.makedirs(generated_product_path, exist_ok=True)
        except Exception as e:
            st.error(f"Failed to create generated images product folder: {e}")
            success = False

    if success:
        st.success(f"Created product folders (input & output) : {safe_product_name} for client {client_folder}")
        return True
    else:
        st.error(f"Failed to create one or both product folders for client {client_folder}!")
        return False

# Function to count images in a folder
def count_images_in_folder(folder_path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(folder_path, ext)))
    return len(image_files)

# Function to display images in a folder with delete buttons
def display_folder_images(folder_path, columns=3):
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(folder_path, ext)))

    if not image_files:
        st.info("이 폴더에 이미지가 없습니다.")
        return
    
    for i, image_path in enumerate(image_files):
        try:
            col1, col2 = st.columns([4, 1])
            with col1:
                img = Image.open(image_path)
                st.image(img, caption=os.path.basename(image_path), use_column_width=True)
            with col2:
                if st.button(f"Delete", key=f"delete_{i}_{os.path.basename(image_path)}"):
                    try:
                        os.remove(image_path)
                        st.success(f"Image {os.path.basename(image_path)} deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting image: {e}")
        except Exception as e:
            st.error(f"Could not load image: {os.path.basename(image_path)}. Error: {e}")

def upload_images_to_folder(folder_path):
    uploaded_files = st.file_uploader("제품 폴더에 업로드할 이미지를 선택하세요",
                                      accept_multiple_files=True,
                                      type=["jpg", "jpeg", "png"],
                                      key=f"uploader_{os.path.basename(folder_path)}")

    if uploaded_files:
        saved_count = 0
        for uploaded_file in uploaded_files:
            try:
                safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in ('.', '_', '-')).rstrip()
                file_path = os.path.join(folder_path, safe_filename)
                
                if os.path.exists(file_path):
                    with open(file_path, "rb") as existing_file:
                        if existing_file.read() == uploaded_file.getvalue():
                            continue  # Skip saving duplicate
                        else:
                            # Modify filename only if contents are different
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
            st.success(f"{saved_count}개 이미지가 업로드되었습니다!")
            # st.rerun()


# Function to create and download a ZIP file
def create_and_download_zip(client_output_path, zip_filename):
    zip_path_base = os.path.join(ROOT_DIR, f"{os.path.basename(client_output_path)}_all_designs_temp")
    
    try:
        if os.path.exists(zip_path_base + ".zip"):
            os.remove(zip_path_base + ".zip")
        with st.spinner("ZIP 파일 생성 중..."):
            shutil.make_archive(
                base_name=zip_path_base,
                format='zip',
                root_dir=client_output_path
            )
            zip_path_full = zip_path_base + ".zip"

            if os.path.exists(zip_path_full):
                with open(zip_path_full, "rb") as f:
                    zip_data = f.read()
                return zip_data
            else:
                st.error("다운로드용 ZIP 파일 생성에 실패했습니다.")
                return None
    except Exception as e:
        st.error(f"다운로드용 ZIP 파일을 생성할 수 없습니다. 오류: {e}")
        st.exception(e)
        return None

# generate_designs function to accept client and product folder
def generate_designs(client_folder, product_folder, generation_method, params):
    if not model_import_success:
        st.error("Model support library could not be imported. Please check the setup.")
        return False
    
    input_folder = os.path.join(INPUTS_DIR, client_folder, product_folder)
    output_folder = os.path.join(OUTPUTS_DIR, client_folder, product_folder)
    generated_folder = os.path.join(GENERATED_IMAGES_DIR, client_folder, product_folder)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(generated_folder, exist_ok=True)

    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    input_images_paths = []
    for ext in image_extensions:
        input_images_paths.extend(glob(os.path.join(input_folder, ext)))

    if not input_images_paths:
        st.error("선택한 클라이언트의 제품 폴더에 입력 이미지가 없습니다!")
        return False

    try:
        model = MultiImageStyleTransfer(model_dir=MODELS_DIR)
    except FileNotFoundError as e:
        st.error(f"모델 파일을 찾을 수 없습니다: {e}")
        st.error(f"Please ensure the 'models' folder exists and contains 'clip-vit-base-patch32' and 'stable-diffusion-2-1' subfolders with model files.")
        return False
    except RuntimeError as e:
        if "no running event loop" in str(e) and sys.version_info >= (3, 12):
            st.error("PyTorch event loop error detected, potentially due to Python 3.12. Consider Python 3.9 or 3.10.")
            return False
        elif "CUDA out of memory" in str(e):
            st.error("GPU 메모리가 부족합니다. 이미지/디자인 수를 줄이거나 다른 GPU 앱을 닫아보세요.")
            return False
        else:
            st.error(f"A PyTorch runtime error occurred: {str(e)}")
            st.exception(e)
            return False
    except Exception as e:
        st.error(f"An unexpected error occurred initializing the model: {str(e)}")
        st.exception(e)
        return False

    try:
        with st.spinner(f"'{product_folder}' 폴더에 디자인을 생성 중입니다... 몇 분 정도 소요될 수 있습니다."):
            variations = []
            start_time = time.time()
            
            # Create progress bar and status text for all methods
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Define num_inference_steps, consistent with what's passed to the model
            num_inference_steps_for_progress = 30

            if generation_method == "표준 방식 (단일 베이스)":
                st.info(f"'{os.path.basename(input_images_paths[0])}' 이미지를 베이스로 하여 {params['num_variations']}개의 변형을 생성 중입니다...")
                
                for i in range(params["num_variations"]):
                    current_variation_num = i + 1
                    status_text.text(f"{current_variation_num}/{params['num_variations']}번째 디자인 생성 중... (단계 0%)")

                    def single_gen_progress_callback(step, timestep, latents):
                        prog = step / num_inference_steps_for_progress
                        overall_progress_value = (i + prog) / params["num_variations"]
                        progress_bar.progress(overall_progress_value)
                        status_text.text(f"{current_variation_num}/{params['num_variations']}번째 디자인 생성 중... (단계 {int(prog*100):02d}%)")

                    variation = model.generate_single_variation(
                        input_images_paths,
                        strength=params["strength"],
                        use_rotation=False,
                        progress_callback=single_gen_progress_callback,
                        num_inference_steps=num_inference_steps_for_progress
                    )
                    if variation:
                        variations.append(variation)
                    progress_bar.progress((i + 1) / params["num_variations"])
                
            elif generation_method == "회전 방식 (다중 베이스)":
                st.info(f"베이스 이미지를 순환하며 {params['num_variations']}개의 변형을 생성 중입니다...")
                
                for i in range(params["num_variations"]):
                    base_idx = i % len(input_images_paths)
                    current_variation_num = i + 1
                    status_text.text(f"{current_variation_num}/{params['num_variations']}번째 디자인 생성 중 (베이스 {base_idx+1})... (단계 0%)")

                    def rotation_gen_progress_callback(step, timestep, latents):
                        prog = step / num_inference_steps_for_progress
                        overall_progress_value = (i + prog) / params["num_variations"]
                        progress_bar.progress(overall_progress_value)
                        status_text.text(f"{current_variation_num}/{params['num_variations']}번째 디자인 생성 중 (베이스 {base_idx+1})... (단계 {int(prog*100):02d}%)")
                    
                    variation = model.generate_single_variation(
                        input_images_paths,
                        strength=params["strength"],
                        use_rotation=True,
                        rotation_index=base_idx,
                        progress_callback=rotation_gen_progress_callback,
                        num_inference_steps=num_inference_steps_for_progress
                    )
                    if variation:
                        variations.append(variation)
                    progress_bar.progress((i + 1) / params["num_variations"])
                
            elif generation_method == "배치 처리 (모든 베이스)":
                st.info(f"{len(input_images_paths)}개의 베이스 이미지 각각에 대해 {params['variations_per_base']}개의 변형을 생성 중입니다...")
                all_variations = []
                num_bases = len(input_images_paths)
                variations_per_base = params["variations_per_base"]
                total_designs_to_generate = num_bases * variations_per_base
                
                generated_designs_count = 0

                for i in range(num_bases):
                    current_base_path = input_images_paths[i]
                    status_text.text(f"{i+1}/{num_bases}번째 베이스 이미지 처리 중: {os.path.basename(current_base_path)} (변형 1/{variations_per_base} - 단계 0%)")
                    ordered_paths = [current_base_path] + [p for j, p in enumerate(input_images_paths) if j != i]

                    def batch_gen_progress_callback(variation_in_batch_idx, total_variations_in_batch, pipe_step, pipe_total_steps, pipe_timestep, pipe_latents):
                        current_variation_prog = pipe_step / pipe_total_steps
                        designs_completed_before_current_one = (i * variations_per_base) + variation_in_batch_idx
                        overall_progress_value = (designs_completed_before_current_one + current_variation_prog) / total_designs_to_generate
                        progress_bar.progress(overall_progress_value)
                        
                        status_text.text(
                            f"베이스 {i+1}/{num_bases} ({os.path.basename(current_base_path)}): 변형 {variation_in_batch_idx+1}/{variations_per_base} (단계 {int(current_variation_prog*100):02d}%)"
                        )

                    base_variations = model.generate_multi_style_variations(
                        ordered_paths,
                        num_variations=variations_per_base,
                        strength=params["strength"],
                        use_rotation=False,
                        progress_callback_per_image=batch_gen_progress_callback,
                        num_inference_steps=num_inference_steps_for_progress
                    )
                    if base_variations:
                        all_variations.extend(base_variations)
                    
                    progress_bar.progress(((i + 1) * variations_per_base) / total_designs_to_generate)

                variations = all_variations
                status_text.text("배치 처리가 완료되었습니다.")

            status_text.text("생성된 디자인을 저장 중입니다...")
            save_progress = st.progress(0)
            saved_count = 0
            
            for i, variation in enumerate(variations):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_filename = f"design_{timestamp}_{i + 1}.png"
                output_path = os.path.join(output_folder, output_filename)
                generated_path = os.path.join(generated_folder, output_filename)
                try:
                    variation.save(output_path)
                    variation.save(generated_path)
                    saved_count += 1
                    save_progress.progress((i + 1) / len(variations))
                except Exception as e:
                    st.error(f"이미지 {output_filename} 저장에 실패했습니다. 오류: {e}")

            end_time = time.time()
            duration = end_time - start_time
            status_text.text(f"디자인 생성 완료! {duration:.2f}초 만에 {saved_count}개의 디자인을 저장했습니다.")

        if saved_count > 0:
            st.success(f"{duration:.2f}초 만에 {saved_count}개의 디자인을 성공적으로 생성하고 '{product_folder}'에 저장했습니다!")
            return True
        else:
            st.error("생성되었거나 저장된 디자인이 없습니다.")
            return False


    except Exception as e:
        st.error(f"An error occurred during design generation: {str(e)}")
        st.error("Details: Check GPU memory (if applicable), input image formats, and file permissions.")
        st.exception(e)
        return False

# Function to display generated images for a specific client/product pair
def display_generated_images(client_name, product_name):
    image_dir = os.path.join(GENERATED_IMAGES_DIR, client_name, product_name)
    if os.path.exists(image_dir):
        images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            st.info(f"No generated images for {client_name}/{product_name}")
            return
            
        for i, img_name in enumerate(images):
            img_path = os.path.join(image_dir, img_name)
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.image(img_path, caption=img_name)
            
            with col2:
                if st.button(f"Delete", key=f"delete_{client_name}_{product_name}_{i}"):
                    try:
                        os.remove(img_path)
                        st.success(f"Image {img_name} deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting image: {e}")
    else:
        st.info(f"No generated images directory for {client_name}/{product_name}")

def main():
    st.title("디자인 생성 인터페이스")

    if sys.version_info >= (3, 12):
        st.warning("Note: Python 3.12 detected. Consider Python 3.9 or 3.10 for potentially better compatibility with some AI libraries.")
    elif sys.version_info < (3, 8):
        st.warning("Note: Python version older than 3.8 detected. Consider updating Python.")

    st.sidebar.header("클라이언트 관리")

    st.sidebar.subheader("새 클라이언트 추가")
    new_client_name = st.sidebar.text_input("새 클라이언트 이름 입력")
    if st.sidebar.button("클라이언트 폴더 생성"):
        if new_client_name:
            if create_client_folder(new_client_name):
                st.rerun()
        else:
            st.sidebar.error("클라이언트 이름을 입력하세요")

    st.sidebar.subheader("클라이언트 선택")
    client_folders = get_client_folders()

    if not client_folders:
        st.info("클라이언트 폴더가 없습니다. 사이드바에서 새로 생성하세요.")
        return

    selected_client_index = st.sidebar.selectbox(
        "클라이언트",
        range(len(client_folders)),
        format_func=lambda i: client_folders[i],
        index=0
    )
    selected_client = client_folders[selected_client_index]

    # Product Folder Management (Combined Input & Output)
    st.sidebar.subheader(f"{selected_client}님의 제품 관리")
    new_product_name = st.sidebar.text_input("새 제품 이름 입력")
    if st.sidebar.button("제품 폴더 생성"):
        if new_product_name:
            if create_product_folder(selected_client, new_product_name):
                st.rerun()
        else:
            st.sidebar.error("제품 이름을 입력하세요")
    
    product_folders = get_product_folders(selected_client, INPUTS_DIR) # Get product folders from the input directory
    if not product_folders:
        st.info(f"{selected_client}님에 대한 제품 폴더가 없습니다. 새로 생성하세요.")
        selected_product = None
    else:
        selected_product = st.sidebar.selectbox(
            "제품 폴더 선택",
            product_folders,
            index=0
        )
        # Store the selected client and product in session state
        st.session_state.selected_client = selected_client
        st.session_state.selected_product = selected_product
        
    client_product_path = os.path.join(INPUTS_DIR, selected_client, selected_product) if selected_product else None

    if client_product_path:
        st.sidebar.subheader(f"{selected_client} / {selected_product}에 이미지 업로드")
        upload_images_to_folder(client_product_path)
    
    st.header(f"클라이언트 작업공간: {selected_client}")
    tab1, tab2, tab3 = st.tabs(["입력 이미지", "생성된 디자인", "생성된 이미지"])

    with tab1:
        st.subheader("입력 이미지")
        if client_product_path and os.path.exists(client_product_path):
            num_images = count_images_in_folder(client_product_path)
            st.write(f"업로드된 이미지 수: {num_images}")
            if num_images > 0:
                display_folder_images(client_product_path)
            else:
                st.info("아직 이미지가 업로드되지 않았습니다. 사이드바에서 이미지를 업로드하세요.")
        else:
            st.info("입력 이미지를 보려면 제품 폴더를 선택하세요.")

    with tab2:
        st.subheader("신규 디자인 생성")
        if not client_product_path or not os.path.exists(client_product_path):
            st.warning("디자인 생성을 위해 입력 이미지가 포함된 제품 폴더를 선택하세요.")
        else:
            num_input_images = count_images_in_folder(client_product_path)
            if num_input_images == 0:
                st.warning("디자인 생성을 위해 최소 하나 이상의 입력 이미지를 업로드하세요.")
            elif not selected_product:
                st.warning("디자인 생성을 위해 제품 폴더를 선택하세요.")
            else:
                generation_method = st.selectbox(
                    "디자인 생성 방식",
                    ["표준 방식 (단일 베이스)", "회전 방식 (다중 베이스)", "배치 처리 (모든 베이스)"],
                    index=0,
                    help="업로드된 이미지들이 구조 및 스타일 참조로 어떻게 사용될지 설정합니다."
                )
                if generation_method == "표준 방식 (단일 베이스)":
                    st.caption("*첫 번째* 이미지를 구조로, *모든* 이미지를 스타일로 사용합니다.")
                elif generation_method == "회전 방식 (다중 베이스)":
                    st.caption("구조로 사용할 이미지를 *회전*하며, 스타일은 *모든* 이미지에서 추출합니다.")
                elif generation_method == "배치 처리 (모든 베이스)":
                    st.caption("각 이미지를 구조로 사용하고, 스타일은 *모든* 이미지에서 추출하여 디자인을 생성합니다.")

                st.subheader("디자인 생성 설정")
                col1, col2 = st.columns(2)
                with col1:
                    strength = st.slider("스타일 전이 강도", 0.1, 0.95, 0.7, 0.05)
                params = {"strength": strength}

                if generation_method in ["표준 방식 (단일 베이스)", "회전 방식 (다중 베이스)"]:
                    with col2:
                        num_variations = st.number_input("디자인 개수", 1, value=3, step=1)
                    params["num_variations"] = num_variations
                elif generation_method == "배치 처리 (모든 베이스)":
                    with col2:
                        variations_per_base = st.number_input("베이스 이미지당 디자인 수", 1, value=2, step=1)
                    params["variations_per_base"] = variations_per_base
                    total_designs_batch = variations_per_base * num_input_images
                    st.caption(f"총 {total_designs_batch}개의 디자인이 생성됩니다.")

                if st.button("✨ 디자인 생성", type="primary", key="generate_button"):
                    if num_input_images > 0 and selected_product:
                        if device == 'cpu':
                            st.warning("CPU에서 실행 중입니다. 속도가 느릴 수 있습니다.")
                        elif num_input_images > 5 or params.get('num_variations', 0) > 5 or params.get('variations_per_base', 0) * num_input_images > 10:
                            st.warning("다수의 이미지 또는 디자인 처리에는 시간이 오래 걸릴 수 있습니다. 잠시 기다려 주세요.")
                        success = generate_designs(selected_client, selected_product, generation_method, params)
                        if success:
                            st.balloons()
                            st.rerun()
                    else:
                        st.error("이미지를 업로드하고 제품 폴더를 선택한 후 디자인을 생성하세요!")


        st.divider()
        st.subheader("이전에 생성된 디자인")
        client_output_base_path = os.path.join(OUTPUTS_DIR, selected_client)
        if os.path.exists(client_output_base_path):
            product_dirs = []
            try:
                product_dirs = sorted([
                    d for d in os.listdir(client_output_base_path)
                    if os.path.isdir(os.path.join(client_output_base_path, d))
                ])
            except Exception as e:
                st.warning(f"제품 폴더를 나열하거나 정렬하는 중 오류 발생: {e}")

            if product_dirs:
                total_designs_across_products = 0
                for product_dir_name in product_dirs:
                    product_dir_path = os.path.join(client_output_base_path, product_dir_name)
                    num_images_in_product = count_images_in_folder(product_dir_path)
                    if num_images_in_product > 0:
                        total_designs_across_products += num_images_in_product
                        st.markdown(f"#### 제품: {product_dir_name} ({num_images_in_product}개 이미지)")
                        display_folder_images(product_dir_path, columns=4)
                        st.divider()
                if total_designs_across_products > 0:
                    st.write(f"모든 제품의 총 생성된 디자인 수: {total_designs_across_products}개")
                    
                    # ZIP 파일 생성은 버튼 클릭 시에만 실행
                    if st.button("ZIP 파일 다운로드 준비", key="prepare_zip"):
                        zip_filename = f"{selected_client}_all_designs.zip"
                        zip_data = create_and_download_zip(client_output_base_path, zip_filename)
                        
                        if zip_data:
                            st.download_button(
                                label=f"모든 디자인 ({total_designs_across_products}개, 전체 제품 포함) ZIP으로 다운로드",
                                data=zip_data,
                                file_name=zip_filename,
                                mime="application/zip",
                                key="download_zip_all"
                            )
                else:
                    st.info("이 클라이언트에 대해 생성된 디자인이 없습니다.")
            else:
                st.info("이 클라이언트에 대해 생성된 디자인이 없습니다.")
        else:
            st.info("이 클라이언트에 대해 생성된 디자인이 없습니다.")
            
    with tab3:
        st.subheader("생성된 이미지")
        if selected_client and selected_product:
            display_generated_images(selected_client, selected_product)
        else:
            st.info("클라이언트와 제품을 선택하세요.")


    st.sidebar.divider()
    st.sidebar.header("사용 방법")
    st.sidebar.info("""
    1.  **클라이언트를 생성하거나 선택하세요.** (사이드바에서 진행)
    2.  **제품 폴더를 생성하세요:** 입력 및 출력 디렉토리에 동일한 폴더가 생성됩니다.
    3.  **제품 폴더를 선택하세요.**
    4.  **선택한 제품 폴더에 이미지를 업로드하세요.** (INPUT)
    5.  **'디자인 생성' 탭으로 이동하세요:** 생성 방식과 파라미터를 설정합니다.
    6.  **'디자인 생성' 버튼을 클릭하세요.** 처리되는 동안 기다려주세요.
    7.  **결과 확인:** 생성된 디자인은 아래에 제품별로 정리되어 표시됩니다. (OUTPUT)
    8.  **다운로드:** '모든 디자인 다운로드' 버튼을 눌러 ZIP 파일로 저장하세요.
    """)
    st.sidebar.divider()
    st.sidebar.caption(f"사용 중인 디바이스: {device.upper()} | Python: {sys.version.split()}")
    st.sidebar.caption(f"모델 로딩 경로: ./models/")

if __name__ == "__main__":
    main()
