# Immunotherapy in Delta-Radionics with Computer Science

 

## 论文1Immunotherapy treatment outcome prediction in metastatic melanoma through an automated multi-objective delta-radiomics model

<center class = "half">
  <img src="Users/wujian/Library/Application Support/typora-user-images/image-20221111101900260.png" alt ="image-20221111101900260.png" style ="zoom:25%">
  <img src="/Users/wujian/Library/Application Support/typora-user-images/image-20221111102155193.png" alt="image-20221111102155193" style="zoom:27.5%;" />

  

该作者做了一个Auto-MODR框架

<center >
  <img src="/Users/wujian/Library/Application Support/typora-user-images/image-20221111103749451.png" alt="image-20221111103749451" style="zoom:25%;" />
  <img src="/Users/wujian/Library/Application Support/typora-user-images/image-20221111103813410.png" alt="image-20221111103813410" style="zoom:45%;" />
</center>



<img src="/Users/wujian/Library/Application Support/typora-user-images/image-20221111105338500.png" alt="image-20221111105338500" style="zoom:50%;" />

几项局限。首先，所有放射组学特征都是基于在单个切片中描绘的二维感兴趣区域计算的；因此，整个转移灶的三维信息丢失了。尽管以前的研究表明，在使用 3D 肿瘤体积信息和 2D 单一代表性切片信息时，机器学习分类器没有统计学上的显着差异 [39-41]，但我们希望在未来的研究中对整个肿瘤进行 3D 分析并调查2D和3D的差异对预测结果的影响。此外，我们选择了本研究中最大的病灶进行分析。根据结果，最大病灶的特征变化有助于更准确地建立预测模型；然而，最大的病灶并不能代表包含黑色素瘤的图像中存在的所有种类和模式。如果我们将来考虑所有病变，预测模型可能会变得更加可靠。其次，我们研究的患者数量相对较少，只有 50 名患者。使用了5折交叉验证才能得到数据。数据集中的患者使用了不同的抑制剂。尽管 PD-1 抑制剂和 CTLA-4 抑制剂作用于 T 细胞活化的不同阶段，但它们是非特异性免疫疗法，对免疫系统具有普遍的刺激作用。在本研究中，我们旨在探索delta-radiomics模型能否有效评估两种免疫疗法的反应，以将模型的应用范围扩展到整体免疫疗法。同时，由于样本量的限制，anti-PD-1和anti-CTLA-4的案例不足以单独建模和分析。因此，我们进行了五折交叉验证实验，然后使用抗 PD-1 病例（较大样本量）作为训练样本和抗 CTLA-4 病例（较小样本量）作为测试样本。

## 论文 2 Using Machine Learning Algorithms to Predict Immunotherapy Response in Patients with Advanced Melanoma

<img src="/Users/wujian/Library/Application Support/typora-user-images/image-20221115170533343.png" alt="image-20221115170533343" style="zoom:50%;" />

用inception v3架构做了两个分类器（分割分类器segmentation classifier和响应分类器response classifier），分割分类器用来根据样本组织内肿瘤区域的分析来预测临床结果，DCNN为每个兴趣类别的每个图块生成一个概率值。对于分割分类器，类别是肿瘤、淋巴细胞和结缔组织区室。对于响应分类器，类是响应和POD（疾病进展progression of disease）。作者将患者ct中每个图块的概率进行平均，将最终概率分配个每个患者，DCNN预测的均方误差被用作准确性的衡量标准。神经网络和逻辑回归分类器计算的概率用于生成ROC曲线。