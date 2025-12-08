# Zde není třeba nic řešit, v políčku dole využijete tento Objekt, který přiřadí anatomickou strukturu daným MNI souřadnicím.
from nilearn import plotting, datasets
import nibabel as nib
import numpy as np
class AtlasLabeler:
    def __init__(self):
        print("Loading Harvard-Oxford Atlases...")
        # Load Cortical Atlas
        self.atlas_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        self.img_cort = self._load_img(self.atlas_cort.maps)
        self.data_cort = self.img_cort.get_fdata()
        self.affine_cort = self.img_cort.affine

        # Load Subcortical Atlas
        self.atlas_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
        self.img_sub = self._load_img(self.atlas_sub.maps)
        self.data_sub = self.img_sub.get_fdata()
        self.affine_sub = self.img_sub.affine

    def _load_img(self, maps):
        if isinstance(maps, str):
            return nib.load(maps)
        return maps

    def __call__(self, coord):
        """Return anatomical structure name from MNI coordinates.
        coord = (R, A, S) in mm
        """
        # 1. Check Cortical Atlas
        label = self._lookup(coord, self.data_cort, self.affine_cort, self.atlas_cort.labels)
        
        # 2. Check Subcortical Atlas (if cortical is background)
        if label == "No label (background)":
            label_sub = self._lookup(coord, self.data_sub, self.affine_sub, self.atlas_sub.labels)
            if label_sub != "No label (background)":
                return label_sub
                
        return label

    def _lookup(self, coord, data, affine, labels):
        # Convert MNI -> voxel indices
        vox = np.linalg.inv(affine).dot(np.append(coord, 1))[:3]
        vox = np.round(vox).astype(int)
        
        # Check bounds
        if np.any(vox < 0) or np.any(vox >= data.shape):
            return "Coordinate outside atlas"
        
        idx = int(data[vox[0], vox[1], vox[2]])
        
        if idx == 0:
            return "No label (background)"
        
        return labels[idx]



if __name__ == "__main__":
    labeler = AtlasLabeler()
    print(labeler((0, 0, 0)))
