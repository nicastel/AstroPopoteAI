import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.web.server.websocket_headers import _get_websocket_headers

from PIL import Image
import time, threading, io, warnings, argparse, json, os
from os import listdir
from importlib import import_module

from pysiril.siril   import *
from pysiril.wrapper import *

from file_queue import FileQueue

# Set image warning and max sizes
WARNING_SIZE = 4096
MAX_SIZE = None
#Image.MAX_IMAGE_PIXELS = 8192

class App:
    def __init__(self):
        self.queue = None
        self.running = True

    def cook(self, file)->str:
        # Show that cooking is starting
        self.info = st.info("Cooking image...", icon="🔥")

        # Set the bar to 0
        bar = st.progress(0)

        app=Siril(R'/usr/bin/siril-cli')

        try:
            cmd=Wrapper(app)    #2. its wrapper
            app.Open()          #2. ...and finally Siril

            #3. Set preferences
            process_dir = '../process'
            cmd.set16bits()
            cmd.setext('fit')

            cmd.load(file.path)
            cmd.save("result.fit")

        except Exception as e :
            st.error("Siril error: " +  str(e), icon="❌")
            return None

        #6. Closing Siril and deleting Siril instance
        app.Close()
        del app

        # Run the process, yield progress
        result = file
        #for i in model.enhance_with_progress(image_rgb, args):
        #    if type(i) == float:
        #        bar.progress(i)
        #    else:
        #        result = i
        #        break

            # Early exit if we are no longer running (user closed the page)
            #if not self.running:
            #    break

        # Clear the bar
        bar.empty()
        return result

    def heart(self):
        # Beacause multiple users may be using the app at once, we need to check if
        # the websocket headers are still valid and to communicate with other threads
        # that we are still "in line"

        while self.running and self.queue.should_run():
            if _get_websocket_headers() is None:
                self.close()
                return

            self.queue.heartbeat()
            time.sleep(1)

    def render(self):
        st.title('AstroPopoteAI')
        st.subheader("Automatically Cook deep space image with AI")

        # Show the file uploader and submit button
        with st.form("my-form", clear_on_submit=True):
            file = st.file_uploader("FILE UPLOADER", type=["fit", "fits","png", "jpg", "jpeg", "tiff"])
            submitted = st.form_submit_button("Cook!")

        if submitted and file is not None:
            image = Image.open(file)

            if image.width > WARNING_SIZE or image.height > WARNING_SIZE:
                st.info("Woah, that image is quite large! You may have to wait a while and/or get unexpected errors!", icon="🕒")

            # Start the queue
            self.queue = FileQueue()
            queue_box = None

            # Wait for the queue to be empty
            while not self.queue.should_run():
                if queue_box is None:
                    queue_box = st.warning("Experincing high demand, you have been placed in a queue! Please wait...", icon ="🚦")
                time.sleep(1)
                self.queue.heartbeat()

            # Start the heart thread while we are upscaling
            t = threading.Thread(target=self.heart)
            add_script_run_ctx(t)
            t.start()

            # Empty the queue box
            if queue_box is not None:
                queue_box.empty()

            # Start the cooking
            a = time.time()
            result = self.cook(file)
            print(f"Cooking took {time.time() - a:.4f} seconds")

            # Check if the cooking failed for whatever reason
            if result is None:
                st.error("Internal error: Cooking failed, please try again later?", icon="❌")
                self.close()
                return

            image = Image.open(result)

            # Empty the info box
            self.info.empty()

            # Large images may take a while to encode
            encoding_prompt = st.info("Cooking complete, encoding...")

            # Convert to bytes
            b = io.BytesIO()
            file_type = file.name.split(".")[-1].upper()
            file_type = "JPEG" if not file_type in ["JPEG", "PNG"] else file_type
            image.save(b, format=file_type)

            # Show success / Download button
            encoding_prompt.empty()
            st.success('Done! Please use the download button to get the highest resolution', icon="🎉")
            st.download_button("Download Full Resolution", b.getvalue(), file.name, "image/" + file_type)

            # Show preview
            image.thumbnail([1024, 1024])
            st.image(image, caption='Image preview', use_column_width=True)

            # Leave the queue for other clients to start upscaling
            self.close()

    def close(self):
        # Exit from queue and stop running
        self.running = False
        if self.queue is not None:
            self.queue.quit()
            self.queue = None

app = App()
app.render()
