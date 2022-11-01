# Manufacturing-Vision-Accelerator-AMD64-v2
Version 2 of the Manufacturing Vision Solution Accelerator includes additional algorithms for object detection, multi-class/label classification and instance segmentation, giving full parity with Azure ML AutoML for Images.  Version 2 also includes support the Azure Computer Vision OCR container.  

The other major difference between v1 and v2 is the use of blob storage for the model repository versus containerizing the model repo.  This flow fits much better into a CI/CD pipeline for MLOps.

Additionally with Version 2, there is a choice between a monolithic CIS module, meaning the capture and inference components are wholly contained within the module, or a modular CIS module, meaning the capture and inference components are shared across all CIS containters.  The modular approach allows for a more atomic approach to any code changes to the camera and/or inference capabilities.

Documentation will be posted soon - please check back!

Hope you find this helpful to your vision analytics pursuits!
