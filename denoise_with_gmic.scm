; This script contains the command syntax for G'MIC git post v1.79
; refer to the original post for scripts for v1.79 & older
; https://discuss.pixls.us/t/a-masashi-wakui-look-with-gimp/2771/6

(define (script-fu-denoise-gmic
            theImage
            baseLayer
        )	
   ; Initialize an undo, so the process can be undone with a single undo
     (gimp-image-undo-group-start theImage)

     (plug-in-gmic-qt 1 theImage baseLayer 1 0 "-v - -ms_patch_smooth 0.8,5,3,5,0,1,1,7,5,4,3,2,1,1,1.3,0") ; call denoiser
     
     ;Ensure the updated image is displayed now
     (gimp-displays-flush)

     (gimp-image-undo-group-end theImage)

) ;end define

(script-fu-register "script-fu-denoise-gmic"
	_"<Image>/Script-Fu/DemoiseGmic..."
            "This script runs the "
            "Nicolas Castel"
            "Nicolas Castel"
            "Marc 2024"
            "*"
	SF-IMAGE		"Image"     0
	SF-DRAWABLE		"Drawable"  0
)
