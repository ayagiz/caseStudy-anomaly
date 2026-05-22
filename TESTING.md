1. TP/TN/FP/FN
Precision
Recall
F1
AUROC

metrikleri kullanıldı. 

- Precision sayesinde false-positive davranışı ölçüldü. Sistemin anomali olarak değerlendirdiği resmin gerçekte normal olması.
- Recall sayesinde false-negative davranışı ölçüldü. Sistemin normal olarak değerlendirdiği resmin gerçekte anomali olması.
- F1 sayesinde precision/recall dengesi ölçüldü.
- AUROC sayesinde threshold dan bağımsız değerlendirme yapıldı.
2. 
Hazelnut results:
    
    Threshold: 4.7 -> normal ve anomali scoreları gözle karşılaştırılıp seçildi

    TP: 58
    TN: 40
    FP: 0
    FN: 12

    Accuracy: 0.8909
    Precision: 1.0000
    Recall: 0.8286
    F1 Score: 0.9062
    AUROC: 0.9868

    Model normal resimleri tespit etmede hiç sorun yaşamadı. Fakat çizik ve kesik bulunan küçük anomalileri normal olarak değerlendirdiği oluyor.

Wood results:

    Threshold: 3.6 -> normal ve anomali scoreları gözle karşılaştırılıp seçildi

    TP: 49
    TN: 19
    FP: 0
    FN: 11

    Accuracy: 0.8608
    Precision: 1.0000
    Recall: 0.8167
    F1 Score: 0.8991
    AUROC: 0.9754

    Sonuçlar Hazelnut ile çok benzer çıktı. Sistem sıvı ve çizik anomalilerini tespit etmede bazen sorun yaşadı. Bazılarını gözle bile ayırt etmesi zor.

3. Patchcore class prediction yerine puan sistemi kullandığı için bir threshold belirlenmesi zaten şarttı. Threshold manual empirical tuning ile karar verildi. Hazelnut ve Wood farklı score table ürettikleri için ayrı threshold lar belirlenmesi gerekti.

4. heatmap sonuçları başarılı ama küçük anomaliler(scratch gibi) zor tespit edilebiliniyor. pixel seviyede gözlem zaten yapılmadı. patchler resimlerde ki detayları azalttı. Pixel seviyede train edilmeside sistem tarafından zaten limitliydi. Patch 100 ken bile RAM yetmedi. Weak localization yapıldı. Bazı normal resimler heatmap de anomali gibi
görünüyordu fakat son sonucu etkilemedi. heatmap her resim için ayrı normalize edildiği için bu gibi sonuçlar normal olarak değerlendirildi.

5. Limitasyonlara rağmen AUROC score oldukça yüksek çıktı 2 cisim türü içinde. Buda sistemin hem shape hem texture oriented resimlerde doğru çalıştığını gösteriyor.

6. Pixel seviyede data train edemediğim için ground truth maskelerini set ini kullanıp IoU, Dice score ve pixel AUROC metrikleri ölçülememiştir.

7. Validation set kullanılmadı çünkü mvtec ad tarafından sağlanmamıştı. Test data sı üzerinden threshold tuning yapıldı.

8. False-Positive 2 cisim içinde 0 sonucu çıktı. 2 cisim içinde conservative behavior sergilendi. Fakat False-Negative hatalar yapıyor sistem. Bazı küçük anomalileri tespit etmede sistem sorun yaşıyor. Anomali olduğu halde normal olarak label lıyor.

