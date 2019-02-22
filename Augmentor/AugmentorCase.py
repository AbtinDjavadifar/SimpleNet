
import Augmentor

WrinklePath="/home/aeroclub/Abtin/CASE/Wrinkle_templates"

p = Augmentor.Pipeline(WrinklePath)
p.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
p.process()

p = Augmentor.Pipeline(WrinklePath)
p.zoom(probability=1, min_factor=1.1, max_factor=1.5)
p.process()

p = Augmentor.Pipeline(WrinklePath)
p.skew_tilt(probability=1)
p.process()

p = Augmentor.Pipeline(WrinklePath)
p.skew_left_right(probability=1)
p.process()

p = Augmentor.Pipeline(WrinklePath)
p.skew_top_bottom(probability=1)
p.process()

p = Augmentor.Pipeline(WrinklePath)
p.skew_corner(probability=1)
p.process()

p = Augmentor.Pipeline(WrinklePath)
p.skew(probability=1)
p.process()

p = Augmentor.Pipeline(WrinklePath)
p.random_distortion(probability=1, grid_width=16, grid_height=16, magnitude=8)
p.process()

p = Augmentor.Pipeline(WrinklePath)
p.shear(probability=1, max_shear_left=20, max_shear_right=20)
p.process()

p = Augmentor.Pipeline(WrinklePath)
p.crop_random(probability=1, percentage_area=0.7)
p.process()

p = Augmentor.Pipeline(WrinklePath)
p.flip_random(probability=1)
p.process()

#p.status
#p.sample(100)
