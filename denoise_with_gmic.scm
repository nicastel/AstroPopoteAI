; This script contains the command syntax for G'MIC git post v1.79
; refer to the original post for scripts for v1.79 & older
; https://discuss.pixls.us/t/a-masashi-wakui-look-with-gimp/2771/6

(define (denoise-gmic 
            filename
        )	
   (let* ((image (car (gimp-file-load RUN-NONINTERACTIVE filename filename)))
          (drawable (car (gimp-image-get-active-layer image))))
     (plug-in-gmic-qt 1 image image 1 0 "-v - ms_patch_smooth 0.8,5,3,5,0,1,1,7,5,4,3,2,1,1,1.3,0") ; call denoiser
     
     (gimp-file-save RUN-NONINTERACTIVE image drawable filename filename)
     (gimp-image-delete image)
)) ;end define

; run via flatpak run org.gimp.GIMP -i -b '(denoise-gmic "foo.tiff")' -b '(gimp-quit 0)'
