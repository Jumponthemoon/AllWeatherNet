
# **AllWeatherNet (ICPR 2024 Best Paper Award)**
**Unified Image Enhancement for Autonomous Driving under Adverse Weather and Lowlight Conditions**

AllWeatherNet is an advanced image enhancement framework designed to improve the visibility of images captured in challenging weather conditions, such as snow, rain, fog, and low-light environments. This solution is particularly tailored for autonomous driving applications, ensuring safety and performance in diverse environments.

![Original Input](https://github.com/Jumponthemoon/AllWeatherNet/assets/39290403/15efd3e4-f878-4295-9e85-6b686d79eddc)
![Original Input 2](https://github.com/Jumponthemoon/AllWeatherNet/assets/39290403/0eb1a130-5ba3-4ed4-bef2-49a4b922e2ff)

## **Key Features**
- **Unified Enhancement:** Enhances images captured under various adverse weather conditions, including snowy, rainy, foggy, and nighttime scenarios.
- **Scaled-Illumination Attention:** Employs a robust scaled-illumination attention mechanism to maintain focus on the road across different conditions.
- **Hierarchical Discrimination:** Utilizes hierarchical patch-level discrimination at scene, object, and texture levels for more effective enhancement.

![Architecture](https://github.com/Jumponthemoon/AllWeatherNet/assets/39290403/0fb128f1-b5c7-4e13-a718-a1254779022a)

## **Environment Setup**

To set up the required environment, run:
```bash
pip install -r requirements.txt
```

## **Dataset Preparation**
1. Download the dataset from the [ACDC official website](https://acdc.vision.ee.ethz.ch/).
2. Organize the dataset in the following structure:
    ```
    ├── ACDC
    │   ├── trainA  # Contains adverse weather images
    │   └── trainB  # Contains normal weather images
    ```

### **Demo Instructions**

1. **Download the Pretrained Model:**  
   Download the pretrained model from [this link](https://drive.google.com/file/d/1n26I1FgwmMtwdKyFZNvd-sDvrR-0qm8v/view?usp=drive_link) and place it in the `checkpoints` folder within the repository.

2. **Set the Demo Image Path:**
   You can put the images to be tested under the folder of `test_data/testA`. Or you can specify the path by setting the `dataroot` variable in `script.py` but your folder should contains `testB` with an image as a placeholder. Your test image can either be the original or a downsampled version from the original dataset.

5. **Run the Script:**  
   Execute the script using the following command:
   ```bash
   python scripts/script.py --predict


## **Acknowledgements**
This project is inspired by [EnlightenGAN](https://github.com/VITA-Group/EnlightenGAN). We greatly appreciate the authors for their outstanding contributions.

## **Citation**
If you find this work useful, please cite:
```bibtex
@inbook{Qian_2024,
   title={AllWeather-Net: Unified Image Enhancement for Autonomous Driving Under Adverse Weather and Low-Light Conditions},
   ISBN={9783031781131},
   ISSN={1611-3349},
   url={http://dx.doi.org/10.1007/978-3-031-78113-1_11},
   DOI={10.1007/978-3-031-78113-1_11},
   booktitle={Pattern Recognition},
   publisher={Springer Nature Switzerland},
   author={Qian, Chenghao and Rezaei, Mahdi and Anwar, Saeed and Li, Wenjing and Hussain, Tanveer and Azarmi, Mohsen and Wang, Wei},
   year={2024},
   month=dec, pages={151–166} }
```

## **To-Do List**
- [x] Release test code
- [ ] Clean and refine training code
- [ ] Add more documentation and tutorials


![Views](https://komarev.com/ghpvc/?username=Jumponthemoon&color=blue)

