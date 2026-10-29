from typing import Tuple, Optional
import math

class DecisionFusionEngine:
    """
    Engine for fusing text and visual forensics probabilities to produce
    a unified verdict, confidence score, and explainable rationale.
    """
    
    def fuse(self, text_prob: Optional[float], visual_prob: Optional[float]) -> Tuple[str, float, str, bool]:
        """
        Fuses multimodal probabilities using late-fusion weighting.
        
        Args:
            text_prob: Probability of text being fake (0.0 to 1.0), or None
            visual_prob: Probability of visual being manipulated (0.0 to 1.0), or None
            
        Returns:
            Tuple containing:
            - verdict (str): Categorical outcome
            - confidence_score (float): Score from 0.0 to 100.0
            - explanation (str): Human-readable rationale
            - cross_modal_conflict (bool): True if severe contradiction detected
        """
        # Determine active modalities and weights
        if text_prob is not None and visual_prob is not None:
            w1, w2 = 0.65, 0.35  # text model is purpose-trained → higher weight
            s = w1 * text_prob + w2 * visual_prob
            
            # Check for contradiction
            contradiction_diff = abs(text_prob - visual_prob)
            cross_modal_conflict = contradiction_diff > 0.4
        elif text_prob is not None:
            s = text_prob
            cross_modal_conflict = False
        elif visual_prob is not None:
            s = visual_prob
            cross_modal_conflict = False
        else:
            # Fallback if no valid modalities
            return "Likely Authentic", 0.0, "No valid data provided for analysis.", False

        # ── THRESHOLD: The TF-IDF model was trained on Western news datasets.
        # Short factual statements often get p_fake in the 0.4–0.65 range
        # simply because they don't resemble the training corpus style.
        # We use a raised threshold of 0.65 so only clearly sensationalist /
        # manipulative writing patterns get flagged as fake.
        FAKE_THRESHOLD = 0.65

        # Calculate confidence relative to the verdict
        if cross_modal_conflict:
            confidence = 1.0 - contradiction_diff
        else:
            confidence = s if s >= FAKE_THRESHOLD else (1.0 - s)
            
        confidence_score = float(confidence * 100.0)
        
        # Base verdict logic
        if cross_modal_conflict:
            verdict = "High Risk Contradiction"
            if text_prob > visual_prob:
                explanation = "Severe contradiction detected: Text claims appear highly suspicious, but visual forensics show no obvious manipulation."
            else:
                explanation = "Severe contradiction detected: Visual forensics indicate severe manipulation, but textual claims appear authentic."
        elif s >= FAKE_THRESHOLD:
            verdict = "Manipulated Media"
            explanation = f"Analysis strongly indicates manipulated content (Overall Score: {confidence_score:.1f}%)."
            if text_prob is not None and text_prob >= 0.7:
                explanation += " NLP models flagged the text as highly deceptive."
            if visual_prob is not None and visual_prob >= 0.7:
                explanation += " Forensic analysis detected significant visual tampering artifacts."
        else:
            verdict = "Likely Authentic"
            explanation = f"Analysis suggests the content is authentic (Overall Score: {confidence_score:.1f}%)."
            if text_prob is not None and text_prob < 0.35:
                explanation += " Text claims align with verified factual patterns."
            if visual_prob is not None and visual_prob < 0.3:
                explanation += " No significant visual manipulation artifacts were detected."
                
        return verdict, confidence_score, explanation, cross_modal_conflict

# Global instance
fusion_service = DecisionFusionEngine()
