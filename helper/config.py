class Config(dict):
    def __init__(self, **kwargs):
        self.checkpoint = ""
        self.init_image_path = ""
        self.input_folder = "./input"
        self.output_folder = "./output"
        self.zoom_image_folder = "./zoom_images"
        self.face_image_folder = "./face_images"
        self.flow_image_folder = "./flow_images"

        self.base_prompt = "(masterpiece, high quality:1.2),"
        self.face_prompt = "(masterpiece, high quality:1.2), face close up,"
        self.neg_prompt = "(worst quality, low quality:1.2), nsfw, nude,"

        self.seed = -1
        self.seed_mode = "fixed"  # fixed,iter,random
        self.sampler_name = "Euler a"
        self.sampler_step = 30
        self.face_sampler_step = 20
        self.cfg_scale = 7
        self.frame_width = 1080
        self.frame_height = 1920
        self.frame_resize = "crop"  # crop,fit,resize
        self.frame_resize_anchor = "center"  # center, top
        self.frame_crop = None  # [x, y, width, height]
        self.frame_zoom = None  # [scale, offsetx, offsety, frames(0=immediately)]
        self.start_frame = 1
        self.dynamic_face_zoom = False

        self.interrogate_model = "clip"  # clip, deepdanbooru
        self.face_interrogate_model = "deepdanbooru"  # clip, deepdanbooru
        self.use_base_img2img = True
        self.use_face_img2img = False
        self.use_zoom_img2img = False

        self.denoising_strength = 0.5
        self.face_denoising_strength = 0.4
        self.face_threshold = 0.35
        self.face_padding = 16
        self.zoom_denoising_strength = 0.65

        self.use_interrogate = False
        self.use_face_interrogate = False

        self.temporalnet_reset_interrogate = False
        self.temporalnet_reset_frames = []
        self.temporalnet = ""  # v1, v2
        self.temporalnet_weight = 0.3
        self.zoom_temporalnet = ""  # v1, v2
        self.zoom_temporalnet_weight = 0.3
        self.face_temporalnet = ""  # v1, v2
        self.face_temporalnet_weight = 0.3
        self.zoom_area_limit = 2048 * 1080
        self.zoom_max_resolusion = 2048
        self.zoom_rects = []  # [x, y, width, height, start_frame, end_frame]
        self.controlnet_lowvram = False
        self.controlnet = []
        self.zoom_controlnet = []
        self.face_controlnet = []
        self.frame_schedule = {}  # { "frame": config }

        for key, value in kwargs.items():
            setattr(self, key, value)


class PatchConfig(dict):
    def __init__(self, **kwargs):
        self.particle_folder = "./input"
        self.bg_folder = "./output"
        self.output_folder = "./patch"
        self.patch_rects = []  # [x, y, width, height, start_frame, end_frame]

        for key, value in kwargs.items():
            setattr(self, key, value)
