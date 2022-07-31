import streamlit as st
from streamlit_option_menu import option_menu
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import shutil

selected_menu = option_menu(
        menu_title = None,
        options = ["Home", "About The Developer", "Contact"],
        icons = ["house", "person-workspace","envelope"],
        menu_icon = "cast",
        orientation = "horizontal"
)

#------------------------------------Page 1 ----------------------------------------
#rad = st.sidebar.radio("Navigation", ["Home", "About The Developer"])

if selected_menu == "Home":
    #################################selecting video ################################
    st.title("My Inception V3 Model")


    predicted_object = st.text_input("Enter the name of the object to be searched")

    predict = predicted_object

    picked_video = st.file_uploader("Select the Video To Be Analyzed")



    if picked_video:
        if st.checkbox("Show Selected Video"):
            st.video(picked_video)

    if st.button("Search"):
        st.success("Search started")
        with st.spinner(text = "Searching, Please wait...?"):

            #######################################################################################    

            ################################# saving video to local folder #######################

            def save_uploaded_file(uploaded_file):

                try:

                    with open(os.path.join('static/videos',uploaded_file.name),'wb') as f:

                        f.write(uploaded_file.getbuffer())

                    return 1    

                except:

                    return 0

            #######################################################################################

            ###################################### chopping video ################################


            #uploaded_file = st.file_uploader("Upload Video")

            if not os.path.exists('static'):
                        os.makedirs('static')

            ##delete if there was a previous search

            if os.path.exists('static/pictures'):
                        shutil.rmtree("static/pictures")


            if not os.path.exists('static/pictures'):
                        os.makedirs('static/pictures')

            if not os.path.exists('static/pictures/selected'):
                        os.makedirs('static/pictures/selected') 
                        
            if not os.path.exists('static/videos'):
                        os.makedirs('static/videos') 

            if picked_video is not None:

                if save_uploaded_file(picked_video):

                    vid            = cv2.VideoCapture(os.path.join('static/videos',picked_video.name))
                    
                    currentframe   = 0
                    
                        
                    while (True):
                        success, frame = vid.read()
                        if success:
                            #cv2.imshow('output', frame)
                            cv2.imwrite(f'static/pictures/frame' + str(currentframe) + '.jpg', frame)


                            currentframe +=1
                        if not success:
                            break
                        
                        
                        #if cv2.waitKey(1) & 0xFF == ord('q'):
                        #    break
                            
                    vid.release()
                    #cv2.destroyAllWindows()

            #######################################################################################  



            ##############################################################################################

            #load the model
            model_inception = load_model('model/Inceptionv3.h5')
            #model_inception = InceptionV3()


            #load an image from file


            for filepath in glob.iglob('static/pictures/*.jpg'):

                real_image = load_img(filepath, target_size=(299, 299))
                        # convert the image pixels to a numpy array
                image = img_to_array(real_image)
                        # reshape data for the model
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                        # prepare the image for the Inception model
                image = preprocess_input(image)
                        # predict the probability across all output classes
                yhat = model_inception.predict(image)
                        # convert the probabilities to class labels
                label = decode_predictions(yhat)
                        # retrieve the most likely result, e.g. highest probability
                label = label[0][0]
                
                        # print the classification
                        #print('%s (%.2f%%)' % (label[1], label[2]*100))

                pred_count = 1

                if label[1] == predict:        
                    src = "%s" % (filepath)
                    dst = "static/pictures/selected"
                    shutil.copy2(src, dst)
                else:
                        #print("no specified object")
                    pass



            #########################################################################################################

            ############################### directories ##############################################################\

            image_path = "static/pictures/selected/*.jpg"
            images_folder = "static/pictures/selected/"
            images_sel    = os.listdir(images_folder)
            selected_images = [cv2.imread(selected_image) for selected_image in glob.glob(image_path)]
            #glob.glob(image_path)

            ############################################################################################################


            #################################################  code for pic plots  ###########################################


            for filepath2 in glob.iglob('static/pictures/selected/*.jpg'):
                for i in range(len(images_sel)):

                    st.image(filepath2, caption = images_sel[i], width = 500)

            if len(images_sel) == 0:
                st.error("Searched item not found in selected the video")



        st.success("Search ended")
        st.balloons()

#-------------------------------------PAGE 2 ----------------------------------------------------

elif selected_menu == "About The Developer":
    #st.title("Who is he?")
    #st.write("The best there is")
    col1, col2 = st.columns(2)


    with col1:
        st.header("Brandon Mutombwa")
        bran_pic = load_img("my_pic/brandon_pic.jpg", target_size=(300, 350))
        st.image(bran_pic, caption = "Tonderai Brandon Mutombwa")
        
    with col2:
        st.write(" ")

    st.write("Brandon is a creative, Data Science student at the University of Zimbabwe who is enthusiastic about executing data driven solutions to increase efficiency, accuracy and utility of internal data processing. He is driven by a strong PASSION AND PURPOSE for solving data problems.")

    
    
 #-------------------------------------PAGE 3 ----------------------------------------------------
elif selected_menu == "Contact":
    st.write("Calls     : +263 776 464 136/ +263 77 586 0625         \nWhatsApp : +263 776 464 136        \nEmail : brandonmutombwa@gmail.com")

    
