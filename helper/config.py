class Config(dict):
    def __init__(self, **kwargs):
        self.host = ""
        self.port = 443
        self.https = True
        self.initial_noise_multiplier = 1
        self.checkpoint = ""
        self.vae = ""
        self.init_image_path = ""
        self.input_folder = "./input"
        self.output_folder = "./output"
        self.source_folder = ""
        self.mask_folder = ""
        self.faceid_folder = ""

        self.base_prompt = "(masterpiece, high quality:1.2),"
        self.base_prompt2 = ""
        self.base_prompt3 = ""
        self.base_prompt4 = ""
        self.base_prompt5 = ""
        self.face_prompt = "(masterpiece, high quality:1.2), face close up,"
        self.neg_prompt = "(worst quality, low quality:1.2), nsfw, nude,"

        self.seed = -1
        self.seed_mode = "fixed"  # fixed,iter,random
        self.subseed = -1
        self.subseed_strength = 0
        self.subseed_mode = "random"  # fixed,iter,random
        self.sampler_name = "Euler a"
        self.sampler_step = 30
        self.cfg_scale = 5
        self.freeu = False
        self.face_sampler_name = ""
        self.face_sampler_step = -1
        self.face_cfg_scale = -1
        self.frame_width = 1080
        self.frame_height = 1920
        self.frame_resize = "crop"  # crop,fit,resize
        self.frame_resize_anchor = "center"  # center, top
        self.frame_crop = None  # [x, y, width, height]
        self.frame_zoom = None  # [scale, offsetx, offsety, frames(0=immediately)]

        self.dynamic_face_zoom = False
        self.dynamic_face_zoom_ratio = 6.5
        self.dynamic_face_zoom_anchor = "center"  # center, left

        self.generate_input_folder = "./input"
        self.generate_output_folder = ""
        self.generate_seed = -1
        self.generate_seed_mode = "fixed"  # fixed,iter,random
        self.generate_sampler_name = ""
        self.generate_sampler_step = -1
        self.generate_width = 512
        self.generate_height = 768
        self.generate_hiresfix = False

        self.interrogate_model = "clip"  # clip, deepdanbooru
        self.face_interrogate_model = "deepdanbooru"  # clip, deepdanbooru
        self.use_base_img2img = True
        self.use_face_img2img = False
        self.use_zoom_img2img = False

        self.denoising_strength = 0.5
        self.face_denoising_strength = 0.4
        self.face_threshold = 0.2
        self.face_padding = 16
        self.face_blur = 16
        self.face_source = "input"  # input, output
        self.face_selection = []
        self.zoom_denoising_strength = 0.4
        self.zoom_blur = 16

        self.use_interrogate = False
        self.use_face_interrogate = False

        self.temporalnet_reset_interrogate = False
        self.temporalnet_reset_frames = []
        self.temporalnet = ""  # v1, v2
        self.temporalnet_weight = 0.3
        self.temporalnet_loopback = True
        self.zoom_temporalnet = ""  # v1, v2
        self.zoom_temporalnet_weight = 0.3
        self.zoom_temporalnet_loopback = True
        self.face_temporalnet = ""  # v1, v2
        self.face_temporalnet_weight = 0.3
        self.face_temporalnet_loopback = True
        self.zoom_area_limit = 2048 * 1080
        self.zoom_max_resolusion = 2048
        self.zoom_rects = []  # [x, y, width, height, start_frame, end_frame]
        self.controlnet_lowvram = False
        self.controlnet = []
        self.zoom_controlnet = []
        self.face_controlnet = []
        self.generate_controlnet = []
        self.frame_schedule = {}  # { "frame": config }

        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.face_sampler_name == "":
            self.face_sampler_name = self.sampler_name
        if self.face_sampler_step == -1:
            self.face_sampler_step = self.sampler_step
        if self.face_cfg_scale == -1:
            self.face_cfg_scale = self.cfg_scale

        if self.generate_output_folder == "":
            self.generate_output_folder = self.generate_input_folder + "_generate"
        if self.generate_sampler_name == "":
            self.generate_sampler_name = self.sampler_name
        if self.generate_sampler_step == -1:
            self.generate_sampler_step = self.sampler_step


class PatchConfig(dict):
    def __init__(self, **kwargs):
        self.particle_folder = "./input"
        self.bg_folder = "./output"
        self.output_folder = "./patch"
        self.patch_rects = []  # [x, y, width, height, start_frame, end_frame]

        for key, value in kwargs.items():
            setattr(self, key, value)
