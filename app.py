import streamlit as st
import numpy as np
import cv2 as cv
from game_utilities import play_game
from filters import *
from object_detection_functions import *


@st.cache_resource
def connect_to_camera(functionality):
    cap=cv.VideoCapture(0)
    return cap
def main_ui():
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 0.5rem;
                    padding-right: 0.5rem;
                }
        </style>
        """, unsafe_allow_html=True)
    st.markdown("""
    <style>
        .st-emotion-cache-10oheav {
            padding: 0rem 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    st.sidebar.title("Activities")
    start_game=st.sidebar.button("PLAY GAME 🔮")
    st.sidebar.header("Functionalities ")
    selections = st.sidebar.selectbox(
        "Select an option",
        (
            '',
            'Filters',
            'Object Detection (Optimized)',
            'Object Detection (Non Optimized)',
            'Green Screen',
            'Invisibility Cloak',
        ),
        on_change=None
    )

    ################filters############################
    if selections == 'Filters':
        uploaded_img = st.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'])
        img_display, filter_display= st.columns(2)
        grayscale_display = st.expander("check the image in Grayscale")
        filter_display.empty()
        img_display.empty()

        filters_options=st.sidebar.radio(
                "Select a filter",
                (
                    'Median',
                    'Mean',
                    'Gaussian',
                    'Laplacian',
                    'Erosion',
                    'Dilate',
                    'seuillage',
                    'Sobel',
                    'Emboss',

                )
            )
        if uploaded_img is None:
            st.warning("Please upload an image")
            
        else:
            img_bytes = uploaded_img.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_cv = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
            if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
                original_img = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
                grayscale_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
                grayscale_display.image(grayscale_img)
            elif len(img_cv.shape) == 2 or (len(img_cv.shape) == 3 and img_cv.shape[2] == 1):
                original_img = img_cv
            if original_img is None:
                st.warning("Please upload another image, we can't read it")
            else:
                img_display.markdown('<p style="text-align: center;"><b>Before</b></p>', unsafe_allow_html=True)
                img_display.image(original_img)
                
                
                
            filter_display.markdown('<p style="text-align: center;"><b>After</b></p>', unsafe_allow_html=True)
            if filters_options == 'Median':
                voisiange = st.sidebar.slider(f'voisinage', 1, 11, 3, 2)
                img_filter = filtre_mediane(original_img,voisiange)
                filter_display.image(img_filter)
            elif filters_options == 'Mean':
                voisiange = st.sidebar.slider(f'voisinage', 1, 11, 3, 2)
                img_filter = filtreMoyen(original_img,voisiange)
                filter_display.image(img_filter)
            elif filters_options == 'Gaussian':
                sigma = st.sidebar.slider(f'sigma', 0.1, 3.0, 1.0, 0.1)
                kernel_size = st.sidebar.slider('Kernel size', 1, 11, 3, 2)
                img_filter = filtre_gauss(original_img,sigma,kernel_size)
                filter_display.image(img_filter)
            elif filters_options == 'Laplacian':
                img_filter = filtre_laplacien(original_img)
                filter_display.image(img_filter)
            elif filters_options == 'Erosion':
                kernel_height = st.sidebar.number_input(f'kernel height', 1, 11, 3, 2)
                kernel_width = st.sidebar.number_input(f'kernel width', 1, 11, 3, 2)
                threshold = st.sidebar.number_input(f'threshold', 0, 255, 127, 1)
                kernel = np.ones((kernel_height, kernel_width), np.uint8)
                img_filter = erosion_filter(original_img, kernel, threshold)
                filter_display.image(img_filter)
            elif filters_options == 'Dilate':
                kernel_height = st.sidebar.number_input(f'kernel height', 1, 11, 3, 2)
                kernel_width = st.sidebar.number_input(f'kernel width', 1, 11, 3, 2)
                threshold = st.sidebar.number_input(f'threshold', 0, 255, 127, 1)
                kernel = np.ones((kernel_height, kernel_width), np.uint8)
                img_filter = dilate_filter(original_img, kernel,threshold)
                filter_display.image(img_filter)
            elif filters_options == 'seuillage':
                seuil = st.sidebar.slider(f'seuil', 0, 255, 127, 1)
                max_value = st.sidebar.slider(f'max_value', 0, 255, 255, 1)
                type = st.sidebar.radio(
                    "Select a type",
                    (
                        '0','1','2','3'
                    )
                )
                img_filter = seuilage(original_img, seuil, max_value, int(type))
                filter_display.image(img_filter)
            elif filters_options == 'Sobel':
                img_filter = filtre_sobel(original_img)
                filter_display.image(img_filter)
            elif filters_options == 'Emboss':
                img_filter = filtre_emboss(original_img)
                filter_display.image(img_filter)
    

    ################################################################# OBJECT DETECTION ########################################################################

    elif selections == 'Object Detection (Optimized)':
        st.header('Color Based Detection🕵️')
        sliders,btn=st.columns([3,1])
        display1,display2 = st.columns(2)
        frame_placeholder = display2.empty()
        mask_placeholder = display1.empty()
        min_blue = st.sidebar.slider(f'min_blue', 0, 255, 69, step=1)
        min_green = st.sidebar.slider(f'min_green', 0, 255, 72, step=1)
        min_red = st.sidebar.slider(f'min_red', 0, 255, 120, step=1)
        max_blue = st.sidebar.slider('max_blue', 0, 255, 0, 1)
        max_green = st.sidebar.slider('max_green', 0, 255, 44, 1)
        max_red = st.sidebar.slider('max_red', 0, 255, 220, 1)
        stop_btn = btn.button('Stop')
        cap = connect_to_camera(selections)
        detect_object_color_camera_optimized(cap,frame_placeholder,mask_placeholder,stop_btn,min_blue, min_green, min_red, max_blue, max_green, max_red)

    elif selections == 'Object Detection (Non Optimized)':
        sliders,btn=st.columns([3,1])
        display1,display2 = st.columns(2)
        min_blue = st.sidebar.slider(f'min_blue', 0, 255, 69, step=1)
        min_green = st.sidebar.slider(f'min_green', 0, 255, 72, step=1)
        min_red = st.sidebar.slider(f'min_red', 0, 255, 120, step=1)
        max_blue = st.sidebar.slider('max_blue', 0, 255, 0, 1)
        max_green = st.sidebar.slider('max_green', 0, 255, 44, 1)
        max_red = st.sidebar.slider('max_red', 0, 255, 220, 1)
        stop_btn = btn.button('Stop')
        frame_placeholder = display2.empty()
        mask_placeholder = display1.empty()
        cap = connect_to_camera(selections)
        detect_object_color_camera(cap,frame_placeholder,mask_placeholder,stop_btn,min_blue, min_green, min_red, max_blue, max_green, max_red)

    elif selections == 'Green Screen':
        st.header('Green Screen')
        sliders,btn=st.columns([3,1])
        display1,display2 = st.columns(2)
        frame_placeholder = display2.empty()
        mask_placeholder = display1.empty()
        min_blue = st.sidebar.slider(f'min_blue', 0, 255, 69, step=1)
        min_green = st.sidebar.slider(f'min_green', 0, 255, 72, step=1)
        min_red = st.sidebar.slider(f'min_red', 0, 255, 120, step=1)
        max_blue = st.sidebar.slider('max_blue', 0, 255, 0, 1)
        max_green = st.sidebar.slider('max_green', 0, 255, 44, 1)
        max_red = st.sidebar.slider('max_red', 0, 255, 220, 1)
        background_image = st.sidebar.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'])
        if background_image is not None:
            img_bytes = background_image.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv.imdecode(nparr, cv.IMREAD_UNCHANGED)
            stop_btn = btn.button('Stop')
            cap = connect_to_camera(selections)
            green_screen(cap,frame_placeholder,mask_placeholder,image,stop_btn,min_blue, min_green, min_red, max_blue, max_green, max_red)
    elif selections== 'Invisibility Cloak':
        st.header('Invisibility Cloak')
        sliders,btn=st.columns([3,1])
        display1,display2 = st.columns(2)
        frame_placeholder = display2.empty()
        mask_placeholder = display1.empty()
        min_blue = st.sidebar.slider(f'min_blue', 0, 255, 69, step=1)
        min_green = st.sidebar.slider(f'min_green', 0, 255, 72, step=1)
        min_red = st.sidebar.slider(f'min_red', 0, 255, 120, step=1)
        max_blue = st.sidebar.slider('max_blue', 0, 255, 0, 1)
        max_green = st.sidebar.slider('max_green', 0, 255, 44, 1)
        max_red = st.sidebar.slider('max_red', 0, 255, 220, 1)
        stop_btn = btn.button('Stop')
        cap = connect_to_camera(selections)
        invisibility_cloak(cap,frame_placeholder,mask_placeholder,stop_btn,min_blue, min_green, min_red, max_blue, max_green, max_red)
###########################################################################Play Game############################################################################
    elif start_game:
        st.subheader('Play Pacman Game , avoid the monsters and win 👾')
        pacman = cv.imread('pacman.png')
        ghost = cv.imread('ghost.png')
        cherry = cv.imread('cherry.png')

        # Resize the images to a suitable size
        pacman_img = cv.resize(pacman, (50, 50))
        ghost_img = cv.resize(ghost, (50, 50))
        cherry_img = cv.resize(cherry, (50, 50))
         # size of the images
        pacman_height, pacman_width= 50,50
        ghost_height, ghost_width = 50,50
        cherry_height, cherry_width = 50,50
        play_game(0.2,0.65,10,3,3,pacman_img,pacman_height,pacman_width,cherry_img,ghost_img,cherry_width,cherry_height,ghost_width,ghost_height)
   

            



        
if __name__ == '__main__':
    main_ui()
     