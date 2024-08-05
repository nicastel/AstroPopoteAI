# AstroPopoteAI

Re-implementation of a service similar to Astro Cooker / FITS Scrubber from Denis Mellican using open source software
"Faire la Popote" means cooking in french

Softwares used in the worfklow :

- ASTAP (cli mode, plate solving more robust than Siril)
- Siril (cli mode, photometric calibration, green noise removal, deconvolution)
- GraXpert (cli mode, Gradient removal AI based)
- Starnet v1 (cli mode, integrated in siril, Stars removal AI based)

TBC not yet implemented :

- Darktable (cli mode, astro denoising, contrast and colors enhencement)
- GIMP (cli mode)
- GMIC (integrated in GIMP)

TBC due to heavy resource usage issues :

- SCUNet (DeNoising AI based)
- AstroSleuth (DeBlurring AI based)

Running via [colab](https://colab.research.google.com/drive/14sydwuVCKFCIw3xLt_x5kE730wQ2kyWN?usp=sharing) : Recommanded way, free gpu accelerated hosting by google

Running locally via docker :

Arm64 (raspberry pi, apple silicon mac) and x64 cpu supported
to run install docker and run the commdands :

docker build -t "astropopoteai:Dockerfile" .

docker run -it docker.io/library/astropopoteai:Dockerfile

Running locally on MacOS
