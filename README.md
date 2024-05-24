# AllWeatherNet:Unified Image enhancement for autonomous driving under adverse weather and lowlight-conditions

## Overview
![ori_input](https://github.com/Jumponthemoon/AllWeatherNet/assets/39290403/15efd3e4-f878-4295-9e85-6b686d79eddc)
![ori_input2](https://github.com/Jumponthemoon/AllWeatherNet/assets/39290403/0eb1a130-5ba3-4ed4-bef2-49a4b922e2ff)

Adverse conditions like snow, rain, nighttime, and fog, pose challenges for autonomous driving perception systems. Existing methods for mitigating the impact of these conditions have limited effectiveness in improving essential computer vision tasks, such as semantic segmentation. Additionally, mainstream methods focus only on one specific condition, for example, removing rain or translating nighttime images into daytime ones. To address these, we propose an image enhancement method to improve the visual quality and clarity degraded by such adverse conditions. Our method utilizes a novel hierarchical architecture named AllWeather-Net to enhance images across all adverse conditions. This architecture incorporates information at three semantic levels: scene, object, and texture, achieved by discriminating patches at each level. Furthermore, we introduce a Scaled Illumination-aware Attention Mechanism (SIAM) that guides the learning towards road elements critical for autonomous driving perception. SIAM exhibits robustness, remaining unaffected by changes in weather conditions or environmental scenes. The proposed AllWeather-Net effectively transforms images into normal weather and daytime scenes, demonstrating superior image enhancement results and subsequently enhancing the performance of semantic segmentation, with up to a 5.3\% improvement in mIoU in the trained domain.  We also show our model's generalization ability by applying it to unseen domains without re-training, achieving up to 3.9 \% mIoU improvement. 
## Features
- Image enhancement for snowy,rainy,foggy and nighttime images within a unified architecture
- A robust scaled-illumination attention remains learning focus on road accross different condition 
- Hierarchical discrimination on different image patch-level regarding: scene,object and texture.

## Installation
```bash
git clone https://github.com/jumponthemoon/AllWeatherNet.git
cd AllWeatherNet
pip install -r requirements.txt
```

## Training
```bash
python scripts/script.py --train

```

## Inference
```bash
python scripts/script.py --predict

```


## Citations

List any references or research papers that your model is based on or uses for implementation.

## Contact

Provide details for users to reach out with questions or collaborations:

- Email: tscq@leeds.ac.uk
- GitHub: [your-github](https://github.com/jumponthemoon)
