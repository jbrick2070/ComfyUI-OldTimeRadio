# Round C -- NVIDIA (nvidia/llama-3.3-nemotron-super-49b-v1.5) elapsed=128.4s

### **1. Agreements and Disagreements with ChatGPT & Gemini**

#### **Agreements:**
- **H1 (Prompt Dilution) is Critical**: Both agree the prompt is overly static and diluted. I concur. The current prompt prioritizes set description over motion, which likely confuses the model into producing static frames.  
- **Negative Prompt Handling**: Both note that `CFG=1.0` nullifies negative prompts, but the *positive* prompt contains negations like "no people," which are risky. Removing these is wise.  
- **Sigma Schedule & Sampler**: Both correctly advise against changing these tonight. The sigma schedule is standardized, and `euler` is stable.  

#### **Disagreements:**
- **H3 (i2v Strength)**:  
  - **ChatGPT**: Recommends lowering to `0.60` to reduce anchoring.  
  - **Gemini**: Argues to keep `0.75` since DMM works with it.  
  - **My Take**: Gemini is more convincing. DMM’s success at `0.75` suggests strength isn’t the root cause. The prompt’s static bias is likely overpowering the motion potential. Lowering strength risks visual drift without guaranteed motion gain.  

- **H2 (Long-Clip Degradation)**:  
  - **ChatGPT**: Downplays length, citing a short static clip (57f) and long dynamic clip (169f).  
  - **Gemini**: Notes LTX was trained on 121f (4.84s @25fps), so 169f is out-of-distribution (OOD), causing freezing.  
  - **My Take**: Gemini’s OOD point is valid. While `stellar_divide` (169f) moved, it may have been lucky. OOD models often freeze to avoid temporal collapse. However, fixing the prompt first is safer than capping length (which breaks Rule C7).  

---

### **2. Factual Errors in Previous Answers**
- **ChatGPT**:  
  - Claimed 700 characters "approach the encoder’s limit" — **False**. T5-XXL has a 512-token window; 700 chars ≈ 150–180 tokens.  
  - Suggested `81 frames` for "4n+1 compatibility" — **False**. LTX uses **8n+1** temporal compression (Gemini corrected this).  

- **Gemini**:  
  - Asserted `169f = 8n+1` (21*8 + 1 = 169) — **Correct**.  
  - Accurately noted LTX’s native training length (121f) — **Correct**.  

---

### **3. Important Omissions by Both**
1. **VAE Temporal Compression Impact**:  
   - LTX’s 8x temporal compression means the model sees fewer latent temporal steps. Long clips (e.g., 169f) may still struggle even if frame counts are 8n+1.  

2. **Prompt Variability Across Clips**:  
   - Why did `stellar_divide` (169f) move while others didn’t? Was its prompt different? If all clips used the same template, this suggests the model is on the edge of failure, and small prompt tweaks could tip it into motion.  

3. **Scene-Cut Spikes in MAD Data**:  
   - High MAD spikes (e.g., 32.78 in `stellar_divide`) suggest the model is cutting scenes rather than animating continuously. The prompt should emphasize "unbroken continuous shot" to mitigate this.  

---

### **4. Prioritized Recommendations**
**Goal**: Unlock motion with minimal changes, respecting Rule C7 (audio sync) and avoiding VRAM optimization.  

1. **Deploy a Minimal Motion-Focused Prompt (Highest Priority)**  
   - **Change**: Replace the 700-character prompt with a **<150-character** motion-centric version.  
   - **Example**:  
     ```text
     Continuous dolly forward. Tuning dial sweeps rhythmically. Amber vacuum tubes pulse. Dust motes drift. Unbroken shot, same console throughout.
     ```  
   - **Rationale**: The i2v anchor (`LTXVImgToVideoConditionOnly`) already provides the visual base. The prompt should focus solely on motion.  
   - **File**: Likely in `prompt_builder.py` or wherever `prompt_text` is constructed.  

2. **Remove Static Descriptions and Negative Language**  
   - **Change**: Strip all static set-dressing terms ("obsidian console," "purple lighting," "film grain") and negations ("no people").  
   - **Rationale**: These dilute motion focus. Use the prompt only for *action*, not appearance.  

3. **Keep `LTX_I2V_STRENGTH = 0.75`**  
   - **Rationale**: DMM works at this value. The issue is prompt structure, not strength. Lowering it risks drift without guaranteed motion.  
   - **File**: `config.py` (e.g., line 20).  

4. **Add "Unbroken Continuous Shot" Language**  
   - **Change**: Explicitly state continuity to reduce scene cuts.  
   - **Example**:  
     ```text
     ... Unbroken continuous take. Same radio console throughout. No cuts.
     ```  

5. **Test with Same Seed for Fair Comparison**  
   - **Action**: Re-render a problematic clip (e.g., `cramped_spaceship_cockpit_humming`) with the new prompt using the **same seed** as the original. Compare MAD scores.  

---

### **5. Uncertainties Requiring Verification**
1. **Why Did `stellar_divide` Work?**  
   - Was its prompt different? If all clips used the same template, this suggests the model is sensitive to small prompt variations.  

2. **Impact of Sigma Schedule on Motion**:  
   - The first five sigmas are clustered near 1.0. Does this throttle motion? Probably not, as DMM uses the same schedule.  

3. **Visual Drift at Lower `LTX_I2V_STRENGTH`**:  
   - If we later lower strength to 0.60, how much drift occurs? Need to test with the new prompt first.  

---

### **Final Verdict**
**Ship the prompt fix tonight.**  
- **Change the prompt to motion-centric, strip static boilerplate, and emphasize continuity.**  
- **Do not alter `LTX_I2V_STRENGTH` or clip length.**  
- **Test with same seed and compare MAD.**  

If motion remains static, revisit `LTX_I2V_STRENGTH` (try 0.65) or consider chunking (but this breaks Rule C7). For now, the prompt is the lowest-risk, highest-reward lever.
