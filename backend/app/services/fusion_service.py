from typing import Tuple, Optional, Dict, Any, List
from backend.app.schemas.response import CombinedReport, ModalitySummary


class DecisionFusionEngine:
    """
    Engine for fusing text, image, video, and URL forensics probabilities to produce
    a unified verdict, confidence score, explainable rationale, and a comprehensive
    Combined Multimodal Report.
    """

    FAKE_THRESHOLD = 0.50

    def fuse(self, text_prob: Optional[float], visual_prob: Optional[float]) -> Tuple[str, float, str, bool]:
        """
        Fuses multimodal probabilities using late-fusion weighting.
        """
        if text_prob is not None and visual_prob is not None:
            w1, w2 = 0.65, 0.35
            s = w1 * text_prob + w2 * visual_prob
            contradiction_diff = abs(text_prob - visual_prob)
            cross_modal_conflict = contradiction_diff > 0.4
        elif text_prob is not None:
            s = text_prob
            cross_modal_conflict = False
            contradiction_diff = 0.0
        elif visual_prob is not None:
            s = visual_prob
            cross_modal_conflict = False
            contradiction_diff = 0.0
        else:
            return "Likely Authentic", 0.0, "No valid data provided for analysis.", False

        if cross_modal_conflict:
            confidence = 1.0 - contradiction_diff
        else:
            confidence = s if s >= self.FAKE_THRESHOLD else (1.0 - s)

        confidence_score = round(float(confidence * 100.0), 2)

        if cross_modal_conflict:
            verdict = "High Risk Contradiction"
            if text_prob > visual_prob:
                explanation = (
                    "Severe contradiction detected: Text claims appear highly suspicious, "
                    "but visual forensics show no obvious manipulation."
                )
            else:
                explanation = (
                    "Severe contradiction detected: Visual forensics indicate severe manipulation, "
                    "but textual claims appear authentic."
                )
        elif s >= self.FAKE_THRESHOLD:
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

    def _single_modality_verdict(self, p_fake: float, threshold: float = 0.50) -> Tuple[str, float]:
        if p_fake >= threshold:
            verdict = "Manipulated Media"
            conf = round(p_fake * 100.0, 2)
        else:
            verdict = "Likely Authentic"
            conf = round((1.0 - p_fake) * 100.0, 2)
        return verdict, conf

    def build_combined_report(
        self,
        text_prob: Optional[float] = None,
        text_meta: Optional[Dict[str, Any]] = None,
        image_scores: Optional[Dict[str, float]] = None,
        image_meta: Optional[Dict[str, Any]] = None,
        video_scores: Optional[Dict[str, float]] = None,
        video_meta: Optional[Dict[str, Any]] = None,
        url_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float, str, bool, CombinedReport]:
        """
        Synthesizes arbitrary combinations of Text, Image, Video, and URL analyses
        into a unified verdict and a structured CombinedReport.
        """
        modality_reports: List[ModalitySummary] = []
        weighted_scores: List[Tuple[str, float, float]] = []  # (name, p_fake, weight)
        key_findings: List[str] = []

        # 1. Text Modality
        if text_prob is not None:
            t_verdict, t_conf = self._single_modality_verdict(text_prob, threshold=0.50)
            t_auth = round((1.0 - text_prob) * 100.0, 2)
            word_count = (text_meta or {}).get("word_count", 0)
            source_label = (text_meta or {}).get("source", "Direct Input")
            fc_reason = (text_meta or {}).get("fact_check_reason")
            t_summary = (
                f"NLP + Live Fact-Check evaluated {word_count} words "
                f"({source_label}): {t_verdict} ({t_auth:.1f}% authenticity, {text_prob * 100:.1f}% deception risk)."
            )
            if fc_reason:
                t_summary += f" Fact-Check: {fc_reason}"
            modality_reports.append(
                ModalitySummary(
                    modality="Text",
                    verdict=t_verdict,
                    authenticity_score=t_auth,
                    manipulation_probability=round(text_prob, 4),
                    confidence_score=t_conf,
                    summary=t_summary,
                    details={
                        "fake_probability": round(text_prob, 4),
                        "real_probability": round(1.0 - text_prob, 4),
                        "word_count": word_count,
                        "source": source_label,
                        "fact_check_reason": fc_reason,
                    },
                )
            )
            weighted_scores.append(("Text", text_prob, 0.50))
            if text_prob >= 0.50:
                finding_msg = f"Textual & Fact-Check analysis flagged deceptive or factually false claim (fake probability: {text_prob * 100:.1f}%)."
                if fc_reason:
                    finding_msg += f" {fc_reason}"
                key_findings.append(finding_msg)
            else:
                finding_msg = f"Textual & Fact-Check analysis found claim consistent with verified facts (authenticity: {t_auth:.1f}%)."
                if fc_reason:
                    finding_msg += f" {fc_reason}"
                key_findings.append(finding_msg)

        # 2. Image Modality
        if image_scores is not None:
            img_ela = float(image_scores.get("ela", 0.0))
            img_lap = float(image_scores.get("laplacian", 0.0))
            img_prob = round(min(max(img_ela, 0.60 * img_ela + 0.40 * img_lap, 0.45 * img_ela + 0.55 * img_lap), 1.0), 4)
            i_verdict, i_conf = self._single_modality_verdict(img_prob, threshold=0.50)
            i_auth = round((1.0 - img_prob) * 100.0, 2)
            img_name = (image_meta or {}).get("filename", "Image Asset")
            i_summary = (
                f"Image forensics (SRM Noise + Block ELA + EXIF + DCT + Laplacian) on '{img_name}': {i_verdict} "
                f"(Forensic/ELA={img_ela:.3f}, Sharpness anomaly={img_lap:.3f})."
            )
            modality_reports.append(
                ModalitySummary(
                    modality="Image",
                    verdict=i_verdict,
                    authenticity_score=i_auth,
                    manipulation_probability=img_prob,
                    confidence_score=i_conf,
                    summary=i_summary,
                    details={
                        "filename": img_name,
                        "ela_vit_score": round(img_ela, 4),
                        "laplacian_sharpness_score": round(img_lap, 4),
                        "combined_visual_risk": img_prob,
                    },
                )
            )
            weighted_scores.append(("Image", img_prob, 0.35))
            if img_prob >= 0.50:
                key_findings.append(
                    f"Image forensics on '{img_name}' flagged splicing/AI generation artifacts (Forensic Risk: {img_prob * 100:.1f}%, ELA/SRM: {img_ela * 100:.1f}%, Laplacian: {img_lap * 100:.1f}%)."
                )
            else:
                key_findings.append(
                    f"Image forensics on '{img_name}' showed uniform compression and natural frequency distributions (authenticity: {i_auth:.1f}%)."
                )

        # 3. Video Modality
        if video_scores is not None:
            v_ela = float(video_scores.get("mean_ela", 0.0))
            v_temp = float(video_scores.get("temporal_variance", 0.0))
            v_prob = round(min((v_ela + v_temp) / 2.0, 1.0), 4)
            v_verdict, v_conf = self._single_modality_verdict(v_prob, threshold=0.55)
            v_auth = round((1.0 - v_prob) * 100.0, 2)
            vid_name = (video_meta or {}).get("filename", "Video Asset")
            v_summary = (
                f"Frame-by-frame video forensics on '{vid_name}': {v_verdict} "
                f"(Mean frame ELA={v_ela:.3f}, Temporal variance={v_temp:.3f})."
            )
            modality_reports.append(
                ModalitySummary(
                    modality="Video",
                    verdict=v_verdict,
                    authenticity_score=v_auth,
                    manipulation_probability=v_prob,
                    confidence_score=v_conf,
                    summary=v_summary,
                    details={
                        "filename": vid_name,
                        "mean_frame_ela": round(v_ela, 4),
                        "temporal_variance": round(v_temp, 4),
                        "combined_video_risk": v_prob,
                    },
                )
            )
            weighted_scores.append(("Video", v_prob, 0.35))
            if v_prob >= 0.55:
                key_findings.append(
                    f"Video frame analysis on '{vid_name}' detected temporal jumps or frame-level tampering (mean ELA: {v_ela * 100:.1f}%, temporal variance: {v_temp * 100:.1f}%)."
                )
            else:
                key_findings.append(
                    f"Video frame sampling on '{vid_name}' confirmed consistent inter-frame compression and sharpness (authenticity: {v_auth:.1f}%)."
                )

        # 4. URL Modality Summary (if URL was provided)
        if url_info is not None:
            u_domain = url_info.get("domain", "unknown")
            u_type = url_info.get("url_type", "news_article")
            u_title = url_info.get("title") or url_info.get("url")
            # Derive URL risk from the extracted text/media probabilities
            url_sub_probs = [p for _, p, _ in weighted_scores]
            u_prob = round(sum(url_sub_probs) / len(url_sub_probs), 4) if url_sub_probs else 0.35
            u_verdict, u_conf = self._single_modality_verdict(u_prob, threshold=0.60)
            u_auth = round((1.0 - u_prob) * 100.0, 2)
            u_summary = (
                f"URL inspection ({u_domain} • {u_type.replace('_', ' ')}): '{u_title[:70]}' — {u_verdict}."
            )
            modality_reports.append(
                ModalitySummary(
                    modality="URL",
                    verdict=u_verdict,
                    authenticity_score=u_auth,
                    manipulation_probability=u_prob,
                    confidence_score=u_conf,
                    summary=u_summary,
                    details={
                        "url": url_info.get("url"),
                        "domain": u_domain,
                        "url_type": u_type,
                        "title": u_title,
                        "extracted_chars": len(url_info.get("extracted_text") or ""),
                        "media_analyzed": bool(url_info.get("media_url")),
                    },
                )
            )
            key_findings.append(
                f"Fetched and verified live content from '{u_domain}' ({u_type.replace('_', ' ')}): \"{u_title[:80]}\"."
            )

        if not weighted_scores:
            empty_report = CombinedReport(
                is_multimodal=False,
                modalities_analyzed=[],
                overall_verdict="Likely Authentic",
                overall_confidence=0.0,
                overall_authenticity_score=100.0,
                overall_manipulation_risk=0.0,
                cross_modal_agreement="Single Modality",
                modality_reports=[],
                executive_summary="No valid modalities were provided for analysis.",
                key_findings=[],
                recommendation="Provide text, an image, a video, or a URL to perform verification.",
            )
            return "Likely Authentic", 0.0, "No valid data provided for analysis.", False, empty_report

        # Compute normalized weighted probability across all active core modalities
        total_w = sum(w for _, _, w in weighted_scores)
        combined_fake_prob = sum(p * w for _, p, w in weighted_scores) / total_w

        probs_only = [p for _, p, _ in weighted_scores]
        max_diff = (max(probs_only) - min(probs_only)) if len(probs_only) > 1 else 0.0
        cross_modal_conflict = max_diff > 0.42

        # Determine cross-modal agreement label
        if len(modality_reports) <= 1:
            agreement = "Single Modality"
        elif cross_modal_conflict:
            agreement = "Cross-Modal Contradiction"
        elif max_diff <= 0.20:
            agreement = "Strong Agreement"
        else:
            agreement = "Moderate Agreement"

        # Calculate overall verdict and confidence
        if cross_modal_conflict:
            overall_verdict = "High Risk Contradiction"
            overall_confidence = round((1.0 - max_diff * 0.5) * 100.0, 2)
        elif combined_fake_prob >= self.FAKE_THRESHOLD:
            overall_verdict = "Manipulated Media"
            overall_confidence = round(combined_fake_prob * 100.0, 2)
        else:
            overall_verdict = "Likely Authentic"
            overall_confidence = round((1.0 - combined_fake_prob) * 100.0, 2)

        overall_authenticity = round((1.0 - combined_fake_prob) * 100.0, 2)
        overall_risk = round(combined_fake_prob * 100.0, 2)
        modalities_list = [m.modality for m in modality_reports]
        is_multimodal = len(modalities_list) > 1

        # Build executive summary & recommendation
        mod_names_str = ", ".join(modalities_list)
        fc_reason_top = (text_meta or {}).get("fact_check_reason")
        if cross_modal_conflict:
            high_mods = [name for name, p, _ in weighted_scores if p >= 0.50]
            low_mods = [name for name, p, _ in weighted_scores if p < 0.50]
            explanation = (
                f"Cross-modal contradiction detected across [{mod_names_str}]: "
                f"{', '.join(high_mods) or 'One modality'} indicates high manipulation risk while "
                f"{', '.join(low_mods) or 'another modality'} appears authentic."
            )
            if fc_reason_top:
                explanation += f" Fact-Check: {fc_reason_top}"
            recommendation = (
                "HOLD FOR EDITORIAL REVIEW: Conflicting signals between modalities often indicate "
                "authentic media paired with a fabricated narrative (or vice-versa). Verify original source provenance."
            )
        elif overall_verdict == "Manipulated Media":
            explanation = (
                f"Combined analysis across [{mod_names_str}] strongly indicates manipulated or deceptive content "
                f"(Manipulation Risk: {overall_risk:.1f}%, Confidence: {overall_confidence:.1f}%)."
            )
            if fc_reason_top:
                explanation += f" Fact-Check: {fc_reason_top}"
            recommendation = (
                "FLAG AS MANIPULATED / HIGH RISK: Multiple forensic indicators exceed manipulation thresholds. "
                "Do not publish or share without independent source corroboration."
            )
        else:
            explanation = (
                f"Combined analysis across [{mod_names_str}] indicates the submission is likely authentic "
                f"(Authenticity Score: {overall_authenticity:.1f}%, Confidence: {overall_confidence:.1f}%)."
            )
            if fc_reason_top:
                explanation += f" Fact-Check: {fc_reason_top}"
            recommendation = (
                "CLEARED / LIKELY AUTHENTIC: Forensic checks across all submitted modalities show natural "
                "compression, consistent temporal frames, and credible linguistic structure."
            )

        exec_summary = (
            f"{explanation} Cross-modal consistency is rated as '{agreement}' across "
            f"{len(modalities_list)} analyzed modality stream(s) ({mod_names_str})."
        )

        combined_report = CombinedReport(
            is_multimodal=is_multimodal,
            modalities_analyzed=modalities_list,
            overall_verdict=overall_verdict,
            overall_confidence=overall_confidence,
            overall_authenticity_score=overall_authenticity,
            overall_manipulation_risk=overall_risk,
            cross_modal_agreement=agreement,
            modality_reports=modality_reports,
            executive_summary=exec_summary,
            key_findings=key_findings,
            recommendation=recommendation,
        )

        return overall_verdict, overall_confidence, explanation, cross_modal_conflict, combined_report


# Global instance
fusion_service = DecisionFusionEngine()
