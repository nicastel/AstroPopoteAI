import sys
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit.web.server.websocket_headers import _get_websocket_headers

import subprocess

from PIL import Image
import time, threading, io, warnings, argparse, json, os
from os import listdir
from importlib import import_module

from pysiril.siril   import *
from pysiril.wrapper import *

from file_queue import FileQueue

import shlex
import logging
import subprocess
from io import StringIO

def run_shell_command(command_line):
    command_line_args = shlex.split(command_line)

    logging.info('Subprocess: "' + command_line + '"')

    try:
        command_line_process = subprocess.Popen(
            command_line_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        process_output, _ =  command_line_process.communicate()

        # process_output is now a string, not a file,
        # you may want to do:
        # process_output = StringIO(process_output)
        print(process_output)
    except (OSError) as exception:
        logging.info('Exception occured: ' + str(exception))
        logging.info('Subprocess failed')
        return False
    else:
        # no exception was raised
        logging.info('Subprocess finished')

    return True

# Set image warning and max sizes
WARNING_SIZE = 4096
MAX_SIZE = None
#Image.MAX_IMAGE_PIXELS = 8192

class App:
    def __init__(self):
        self.queue = None
        self.running = True

    def cook(self, file:UploadedFile)->str:
        # Show that cooking is starting
        self.info = st.info("Cooking image...", icon="🔥")

        bytes = file.getvalue()

        # Set the bar to 0
        bar = st.progress(0)

        filename = "/tmp/"+file.name.replace(' ', '_').lower()
        # write the upldoaed file on the disk
        # Open in "wb" mode to
        # write a new file, or
        # "ab" mode to append
        with open(filename, "wb") as binary_file:
            # Write bytes to file
            binary_file.write(bytes)
            binary_file.flush()
            binary_file.close()

            file.close()
            del binary_file
        del file

        # Set the bar to 5
        bar.progress(5)

        imageLocation = st.empty()

        convert = st.info("Convert to fit / debayer with siril...", icon="🕒")

        siril_app=Siril(R'/usr/bin/siril-cli')

        try:
            cmd=Wrapper(siril_app)    #2. its wrapper
            siril_app.Open()          #2. ...and finally Siril

            #3. Set preferences

            cmd.setext('fits')
            cmd.set("core.catalogue_namedstars=/app/namedstars.dat")
            cmd.set("core.catalogue_unnamedstars=/app/unnamedstars.dat")
            cmd.set("core.catalogue_tycho2=/app/deepstars.dat")
            cmd.set("core.catalogue_nomad=/app/USNO-NOMAD-1e8.dat")
            cmd.set("core.starnet_exe=/app/run_starnet.sh")

            # convert to fit / debayer to start from a debayered fit file
            cmd.cd("/tmp/")
            cmd.convert("light",debayer=True)
            os.remove(filename)
            cmd.load("light_00001.fits")
            cmd.autostretch()
            cmd.savejpg("/tmp/preview")
            cmd.close()

            image = Image.open("/tmp/preview.jpg")
            # Show preview
            image.thumbnail([1024, 1024])
            imageLocation.image(image, caption='Image preview', use_column_width=True)

            bar.progress(10)
            convert.info("Convert to fit / debayer with siril", icon="✅")

            platesolve = st.info("Plate solving with astap...", icon="🕒")

            # 1st Step : plate solving with astap
            run_shell_command("/app/astap_cli -f /tmp/light_00001.fits -update")

            bar.progress(20)
            platesolve.info("Plate solving with astap", icon="✅")

            # 2nd Step : gradient removal with graxpert
            gradient = st.info("Remove gradient with graXpert...", icon="🕒")
            os.chdir("/app/GraXpert-3.0.2")
            run_shell_command("/opt/venv/bin/python3 -m graxpert.main /tmp/light_00001.fits -cli")
            bar.progress(30)
            gradient.info("Remove gradient with graXpert", icon="✅")

            # 3rd Step : various processing with Siril
            # photometric calibration
            # green noise removal
            # auto stretch
            # star desaturation
            # deconvolution

            cmd.load("light_00001_GraXpert.fits")
            cmd.autostretch()

            cmd.savejpg("/tmp/preview")

            image.close()
            image = Image.open("/tmp/preview.jpg")
            # Show preview
            image.thumbnail([1024, 1024])
            imageLocation.image(image, caption='Image preview', use_column_width=True)

            cmd.load("light_00001_GraXpert.fits")
            photometric = st.info("Photometric calibration with siril...", icon="🕒")
            cmd.pcc()
            cmd.rmgreen()
            cmd.save("light_00001_GraXpert_pcc_green")
            cmd.autostretch()
            cmd.savejpg("/tmp/preview")

            image.close()
            image = Image.open("/tmp/preview.jpg")
            # Show preview
            image.thumbnail([1024, 1024])
            imageLocation.image(image, caption='Image preview', use_column_width=True)

            bar.progress(40)
            photometric.info("Photometric calibration with siril", icon="✅")

            deconvol = st.info("Apply deconvolution with siril...", icon="🕒")
            cmd.load("light_00001_GraXpert_pcc_green.fits")
            cmd.unclipstars()
            cmd.Execute("makepsf stars")
            cmd.Execute("rl")
            cmd.save("light_00001_GraXpert_pcc_green_deconvol")


            stretch = st.info("Auto stretch with siril...", icon="🕒")
            cmd.autostretch()
            cmd.savejpg("/tmp/preview")

            image.close()
            image = Image.open("/tmp/preview.jpg")
            # Show preview
            image.thumbnail([1024, 1024])
            imageLocation.image(image, caption='Image preview', use_column_width=True)
            bar.progress(50)
            deconvol.info("Apply deconvolution with siril", icon="✅")
            bar.progress(60)
            stretch.info("Auto stretch with siril", icon="✅")
            #cmd.savetif("/app/result")

            # 4th Step : stars removal with starnet v1
            stars = st.info("Remove stars with starnet v1...", icon="🕒")
            cmd.starnet()
            bar.progress(70)
            cmd.save("/tmp/starless")
            cmd.savetif("/tmp/starless",astro=True)
            cmd.savepng("/tmp/starless")
            stars.info("Remove stars with starnet v1", icon="✅")

            # 5th Step : starless denoising with GraXpert
            denoise = st.info("Denoise with GraXpert...", icon="🕒")
            os.chdir("/app/GraXpert-3.0.2")
            run_shell_command("/opt/venv/bin/python3 -m graxpert.main /tmp/starless.fits -cli -cmd denoising")
            cmd.load("starless_GraXpert.fits")
            bar.progress(80)
            denoise.info("Denoise with GraXpert...", icon="✅")

            # 6th Step : colors/contrast enhancements with darktable
            #darktable = st.info("Denoise and enhance colors and contrast of starless with darktable...", icon="🕒")
            #run_shell_command("darktable-cli /tmp/starless.png /tmp/out.tif --style astro --verbose --core --configdir /root/.config/darktable/")
            #cmd.load("out.tif")
            #bar.progress(80)
            #darktable.info("Denoise and enhance colors and contrast of starless with darktable...", icon="✅")

            # save finals files
            cmd.save("/app/result")
            cmd.savejpg("/app/result")

            # clean up
            os.remove("/tmp/light_00001_GraXpert.fits")
            os.remove("/tmp/light_00001.fits")

        except Exception as e :
            st.error("Siril error: " +  str(e), icon="❌")
            return None

        #6. Closing Siril and deleting Siril instance
        siril_app.Close()
        del siril_app

        # Set the bar to 50
        bar.progress(50)

        # 4th Step : star removal with Starnet

        # 5th Step : astro denoising with darktable on the starless

        # Run the process, yield progress
        result = "/app/result.jpg"
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
        st.subheader("Automatically Cook deep space images with open source tools")

        # Show the file uploader and submit button
        with st.form("my-form", clear_on_submit=True):
            file = st.file_uploader("FILE UPLOADER", type=["fit", "fits","png", "jpg", "jpeg", "tiff"])
            submitted = st.form_submit_button("Cook!")

        if submitted and file is not None:
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
            st.download_button("Download Full Resolution", b.getvalue(), "result.jpg", "image/jpeg")

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
