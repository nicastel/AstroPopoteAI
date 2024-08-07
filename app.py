import sys
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.uploaded_file_manager import UploadedFile

import subprocess

from PIL import Image
import time, threading, io, warnings, argparse, json, os, glob
from os import listdir
from importlib import import_module

from pysiril.siril   import *
from pysiril.wrapper import *

from file_queue import FileQueue

import shlex
import logging
import subprocess
from io import StringIO

from spandrel import ImageModelDescriptor, ModelLoader
import torch, cv2
import numpy as np

@st.fragment
def download_button_no_refresh(text, file_content, filename, type):
    st.download_button(text, file_content, filename, type)

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

# suppported for SCUNet : Nvidia GPU / Apple MPS / DirectML on Windows / CPU
def get_device() -> torch.device:
    if os.name == 'nt':
        import torch_directml
        return torch_directml.default_device()
    else:
        return torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] == 1:
        pass
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img)
    return tensor.unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.cpu().squeeze().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = np.clip((image * 255.0).round(), 0, 255)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def image_inference_tensor(
    model: ImageModelDescriptor, tensor: torch.Tensor
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(tensor)

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
            cmd.set("core.catalogue_namedstars=/content/AstroPopoteAI/namedstars.dat")
            cmd.set("core.catalogue_unnamedstars=/content/AstroPopoteAI/unnamedstars.dat")
            cmd.set("core.catalogue_tycho2=/content/AstroPopoteAI/deepstars.dat")
            cmd.set("core.catalogue_nomad=/content/AstroPopoteAI/USNO-NOMAD-1e8.dat")
            cmd.set("core.starnet_exe=/content/AstroPopoteAI/run_starnet.sh")

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
            run_shell_command("/content/AstroPopoteAI/astap_cli -f /tmp/light_00001.fits -update")

            bar.progress(20)
            platesolve.info("Plate solving with astap", icon="✅")

            # 2nd Step : gradient removal with graxpert
            gradient = st.info("Remove gradient with graXpert...", icon="🕒")
            os.chdir("/content/AstroPopoteAI/GraXpert-3.0.2")
            run_shell_command("python3 -m graxpert.main /tmp/light_00001.fits -cli")
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
            #cmd.rmgreen() => alter the color too much
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

            bar.progress(50)
            deconvol.info("Apply deconvolution with siril", icon="✅")

            stretch = st.info("Auto stretch with siril...", icon="🕒")
            cmd.autostretch()
            cmd.savejpg("/tmp/preview")

            image.close()
            image = Image.open("/tmp/preview.jpg")
            # Show preview
            image.thumbnail([1024, 1024])
            imageLocation.image(image, caption='Image preview', use_column_width=True)

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
            # commented => the result are muchbetter with SCUNet without big performance penalty on colab
            #denoise = st.info("Denoise with GraXpert...", icon="🕒")
            #os.chdir("/content/AstroPopoteAI/GraXpert-3.0.2")
            #run_shell_command("python3 -m graxpert.main /tmp/starless.fits -cli -cmd denoising")
            #cmd.load("starless_GraXpert.fits")
            #bar.progress(80)
            #denoise.info("Denoise with GraXpert...", icon="✅")

            # 5th Step : Denoising via SCUNet
            # initialize the main device : GPU, or CPU
            denoise = st.info("Denoise with SCUNet...", icon="🕒")
            device = get_device()

            # load a model from disk
            model = ModelLoader().load_from_file(r"/content/AstroPopoteAI/scunet_color_real_psnr.pth")
            # make sure it's an image to image model
            assert isinstance(model, ImageModelDescriptor)
            # send it to the GPU and put it in inference mode
            model.to(device).eval()

            # read image out send it to the GPU
            imagecv2in = cv2.imread(str("/tmp/starless.tif"), cv2.IMREAD_COLOR)
            tensorin = image_to_tensor(imagecv2in).to(device)

            # process the image and write it to the disk
            imagecv2out = tensor_to_image(image_inference_tensor(model, tensorin))
            cv2.imwrite(str("/tmp/starless_denoised.tif"), imagecv2out)

            cmd.load("starless_denoised.tif")

            bar.progress(80)
            denoise.info("Denoise with SCUNet...", icon="✅")

            # 6th Step : colors/contrast enhancements with darktable
            #darktable = st.info("Denoise and enhance colors and contrast of starless with darktable...", icon="🕒")
            #run_shell_command("darktable-cli /tmp/starless.png /tmp/out.tif --style astro --verbose --core --configdir /root/.config/darktable/")
            #cmd.load("out.tif")
            #bar.progress(80)
            #darktable.info("Denoise and enhance colors and contrast of starless with darktable...", icon="✅")

            # save finals files
            cmd.save("/content/AstroPopoteAI/result_starless")
            cmd.savejpg("/content/AstroPopoteAI/result_starless")

            cmd.load("/tmp/starmask_light_00001_GraXpert_pcc_green.fits")
            cmd.gauss(1.2)
            cmd.unsharp(2,1)
            cmd.save("/content/AstroPopoteAI/result_starmask")
            cmd.savejpg("/content/AstroPopoteAI/result_starmask")

            # 7th Step : final image recomposition with Siril
            # Combine the starless and starmask images using pixel math.
            cmd.pm("$result_starmask$ + 0.6* $result_starless$")
            cmd.clahe(2.0,8)
            cmd.satu(1,0)
            cmd.save("/content/AstroPopoteAI/result_final")
            cmd.savejpg("/content/AstroPopoteAI/result_final")

            # clean up
            for f in glob.glob("/tmp/*.fits"):
                os.remove(f)
            for f in glob.glob("/tmp/*.jpg"):
                os.remove(f)
            for f in glob.glob("/tmp/light*"):
                os.remove(f)
            for f in glob.glob("/tmp/starless*"):
                os.remove(f)

        except Exception as e :
            st.error("Siril error: " +  str(e), icon="❌")
            return None

        #6. Closing Siril and deleting Siril instance
        siril_app.Close()
        del siril_app

        # Run the process, yield progress
        result = "/content/AstroPopoteAI/result"

        # Clear the bar
        bar.empty()
        return result

    def heart(self):
        # Beacause multiple users may be using the app at once, we need to check if
        # the websocket headers are still valid and to communicate with other threads
        # that we are still "in line"

        while self.running and self.queue.should_run():
            if st.context.headers is None:
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

            # Empty the info box
            self.info.empty()

            # Large images may take a while to encode
            encoding_prompt = st.info("Cooking complete, encoding...")

            image_starless = Image.open(result+"_starless.jpg")

            # Convert to bytes
            starless_b = io.BytesIO()
            file_type = "JPEG"
            image_starless.save(starless_b, format=file_type)

            image_starmask = Image.open(result+"_starmask.jpg")

            # Convert to bytes
            starmask_b = io.BytesIO()
            file_type = "JPEG"
            image_starmask.save(starmask_b, format=file_type)

            # Show success / Download button
            encoding_prompt.empty()
            st.success('Done! Please use the downloads button to get the highest resolution', icon="🎉")

            download_button_no_refresh("Download Starless Full Resolution", starless_b.getvalue(), "result_starless.jpg", "image/jpeg")

            download_button_no_refresh("Download Starmask Full Resolution", starmask_b.getvalue(), "result_starmask.jpg", "image/jpeg")

            # Show preview
            image_starless.thumbnail([1024, 1024])
            st.image(image_starless, caption='Image preview', use_column_width=True)

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
