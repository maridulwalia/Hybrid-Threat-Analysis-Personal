"""
Phase 10 Stage B: Document Parsing and Section-Aware Chunking.
Parses authoritative raw snapshots in artifacts/rag/raw/:
  1. MITRE ATT&CK v20.0 STIX 2.1 JSON (enterprise-attack-20.0.json)
  2. NIST SP 800-61 Revision 3 PDF (NIST.SP.800-61r3.pdf)
  3. NVD API 2.0 CVE-2014-0160 JSON (nvd_cve_2014_0160.json)
  4. OWASP Top 10 2021 Markdown files (A01, A02, A03, A05, A07)

Implements the pre-registered chunking grid:
  - C256: 256 words, overlap 50 words
  - C512: 512 words, overlap 100 words (Pre-registered Baseline)
  - C1024: 1024 words, overlap 200 words

Writes:
  - artifacts/rag/chunks_c256_v1.json
  - artifacts/rag/chunks_c512_v1.json
  - artifacts/rag/chunks_c1024_v1.json
"""

import json
import re
import time
from pathlib import Path
from pypdf import PdfReader

def sliding_window_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Splits text into overlapping chunks of words."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size // 2
        
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks

def parse_mitre_stix(stix_path: Path) -> list[dict]:
    print(f"[Parser] Reading MITRE ATT&CK STIX from {stix_path}...")
    with open(stix_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    objects = data.get("objects", [])
    
    # Index course of action (mitigations) by id
    mitigations = {}
    for obj in objects:
        if obj.get("type") == "course-of-action":
            mitigations[obj.get("id")] = obj
            
    # Index detection strategies / analytics
    detection_items = {}
    for obj in objects:
        if obj.get("type") in ["x-mitre-detection-strategy", "x-mitre-analytic", "x-mitre-data-component"]:
            detection_items[obj.get("id")] = obj
            
    # Relationships mapping: attack-pattern -> mitigations & detections
    tech_mitigations = {}
    tech_detections = {}
    for obj in objects:
        if obj.get("type") == "relationship":
            rel_type = obj.get("relationship_type")
            src_ref = obj.get("source_ref", "")
            tgt_ref = obj.get("target_ref", "")
            rel_desc = obj.get("description", "").strip()
            
            if rel_type == "mitigates" and tgt_ref.startswith("attack-pattern--"):
                mit_obj = mitigations.get(src_ref)
                mit_name = mit_obj.get("name", "Unknown Mitigation") if mit_obj else "Mitigation"
                tech_mitigations.setdefault(tgt_ref, []).append(f"- **{mit_name}**: {rel_desc}")
            elif rel_type == "detects" and tgt_ref.startswith("attack-pattern--"):
                det_obj = detection_items.get(src_ref)
                det_name = det_obj.get("name", "Detection Analytic") if det_obj else "Detection"
                tech_detections.setdefault(tgt_ref, []).append(f"- **{det_name}**: {rel_desc}")

    # Extract enterprise attack patterns
    parsed_sections = []
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue
            
        ext_refs = obj.get("external_references", [])
        mitre_ref = [r for r in ext_refs if r.get("source_name") in ["mitre-attack", "mitre-enterprise-attack"]]
        if not mitre_ref:
            continue
            
        tech_id = mitre_ref[0].get("external_id")
        tech_name = obj.get("name", "")
        tech_stix_id = obj.get("id")
        
        # Tactics
        tactics = [p.get("phase_name", "") for p in obj.get("kill_chain_phases", []) if p.get("kill_chain_name") == "mitre-attack"]
        tactic_str = ", ".join(tactics) if tactics else "Enterprise"
        
        # 1. Description Section
        desc_text = obj.get("description", "").strip()
        if desc_text:
            parsed_sections.append({
                "source": "MITRE ATT&CK v20.0",
                "technique_id": tech_id,
                "technique_name": tech_name,
                "tactics": tactic_str,
                "section_type": "description",
                "title": f"{tech_id}: {tech_name} ({tactic_str})",
                "text": f"MITRE ATT&CK Technique {tech_id}: {tech_name}\nTactics: {tactic_str}\n\nDescription:\n{desc_text}"
            })
            
        # 2. Mitigations Section
        mits = tech_mitigations.get(tech_stix_id, [])
        if mits:
            mit_content = "\n".join(mits)
            parsed_sections.append({
                "source": "MITRE ATT&CK v20.0",
                "technique_id": tech_id,
                "technique_name": tech_name,
                "tactics": tactic_str,
                "section_type": "mitigation",
                "title": f"{tech_id}: {tech_name} - Mitigations",
                "text": f"MITRE ATT&CK Technique {tech_id}: {tech_name}\nMitigation Guidance:\n{mit_content}"
            })
            
        # 3. Detection Strategies Section
        dets = tech_detections.get(tech_stix_id, [])
        if dets:
            det_content = "\n".join(dets)
            parsed_sections.append({
                "source": "MITRE ATT&CK v20.0",
                "technique_id": tech_id,
                "technique_name": tech_name,
                "tactics": tactic_str,
                "section_type": "detection",
                "title": f"{tech_id}: {tech_name} - Detection Analytics",
                "text": f"MITRE ATT&CK Technique {tech_id}: {tech_name}\nDetection Analytics:\n{det_content}"
            })
            
    print(f"[Parser] Extracted {len(parsed_sections)} structured technique sections from MITRE ATT&CK.")
    return parsed_sections

def parse_nist_pdf(pdf_path: Path) -> list[dict]:
    print(f"[Parser] Reading NIST SP 800-61r3 PDF from {pdf_path}...")
    reader = PdfReader(str(pdf_path))
    
    sections = []
    full_text_pages = []
    for i, page in enumerate(reader.pages):
        p_text = page.extract_text()
        if p_text:
            # Clean headers / page numbering
            p_text = re.sub(r'NIST SP 800-61r3\s+Incident Response Recommendations[^\n]*', '', p_text)
            full_text_pages.append((i + 1, p_text.strip()))
            
    # Group pages into thematic sections:
    # Pages 1-8: Executive Summary & Overview
    # Pages 9-22: CSF 2.0 Community Profile - Detect (DE) & Incident Detection
    # Pages 23-38: CSF 2.0 Community Profile - Respond (RS) & Analysis/Mitigation
    # Pages 39-48: CSF 2.0 Community Profile - Recover (RC) & Continuous Improvement
    
    sec_map = [
        ("NIST-CSF-OVERVIEW", "Executive Summary & Community Profile Framework", 1, 8, "framework_overview"),
        ("NIST-CSF-DE", "Detect Function (DE): Continuous Anomaly & Threat Detection", 9, 22, "detection"),
        ("NIST-CSF-RS", "Respond Function (RS): Incident Analysis (RS.AN) & Incident Mitigation (RS.MI)", 23, 38, "incident_response"),
        ("NIST-CSF-RC", "Recover Function (RC): Recovery, Eradication & Post-Incident Activity", 39, 48, "recovery"),
    ]
    
    for tech_id, title, start_p, end_p, sec_type in sec_map:
        pg_texts = [txt for p_num, txt in full_text_pages if start_p <= p_num <= end_p]
        combined = "\n\n".join(pg_texts)
        if combined.strip():
            sections.append({
                "source": "NIST SP 800-61r3",
                "technique_id": tech_id,
                "technique_name": title,
                "tactics": "Incident Management",
                "section_type": sec_type,
                "title": f"NIST SP 800-61r3: {title} (Pages {start_p}-{end_p})",
                "text": f"NIST SP 800-61 Revision 3: {title}\nPages {start_p}-{end_p}\n\n{combined}"
            })
            
    print(f"[Parser] Extracted {len(sections)} major CSF functions from NIST SP 800-61r3.")
    return sections

def parse_nvd_json(nvd_path: Path) -> list[dict]:
    print(f"[Parser] Reading NVD CVE-2014-0160 from {nvd_path}...")
    with open(nvd_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    cve = data["vulnerabilities"][0]["cve"]
    cve_id = cve["id"]
    desc = cve["descriptions"][0]["value"]
    
    # CVSS metrics
    metrics = cve.get("metrics", {})
    cvss_v2 = metrics.get("cvssMetricV2", [{}])[0].get("cvssData", {})
    base_score = cvss_v2.get("baseScore", 5.0)
    access_vector = cvss_v2.get("accessVector", "NETWORK")
    
    vuln_text = (
        f"Vulnerability Record {cve_id}: OpenSSL TLS Heartbeat Extension Memory Disclosure (Heartbleed)\n"
        f"Base Severity Score: {base_score} (Access Vector: {access_vector})\n\n"
        f"Description:\n{desc}\n\n"
        f"Remediation & Mitigation Guidance:\n"
        f"Upgrade OpenSSL packages to version 1.0.1g or higher. Alternatively, recompile OpenSSL libraries with "
        f"the flag -DOPENSSL_NO_HEARTBEATS to eliminate the vulnerable code path. Revoke and re-issue all SSL/TLS private "
        f"keys and certificates that were deployed on vulnerable servers, as private keys could have been compromised "
        f"without detection in network logs. Implement network intrusion detection signatures (Snort/Suricata) matching "
        f"inconsistent TLS heartbeat payload length fields."
    )
    
    sections = [{
        "source": "NVD API 2.0",
        "technique_id": "CVE-2014-0160",
        "technique_name": "OpenSSL Heartbleed Vulnerability (CVE-2014-0160)",
        "tactics": "Exploitation of Public-Facing Application",
        "section_type": "vulnerability_advisory",
        "title": "NVD CVE-2014-0160: OpenSSL TLS Heartbeat Information Disclosure",
        "text": vuln_text
    }]
    print(f"[Parser] Extracted {len(sections)} vulnerability record from NVD.")
    return sections

def parse_owasp_docs(owasp_dir: Path) -> list[dict]:
    print(f"[Parser] Reading OWASP Top 10 documents from {owasp_dir}...")
    owasp_mapping = {
        "A01_2021-Broken_Access_Control.md": ("OWASP-2021-A01", "A01:2021 Broken Access Control"),
        "A02_2021-Cryptographic_Failures.md": ("OWASP-2021-A02", "A02:2021 Cryptographic Failures"),
        "A03_2021-Injection.md": ("OWASP-2021-A03", "A03:2021 Injection (SQL / XSS / Command Injection)"),
        "A05_2021-Security_Misconfiguration.md": ("OWASP-2021-A05", "A05:2021 Security Misconfiguration"),
        "A07_2021-Identification_and_Authentication_Failures.md": ("OWASP-2021-A07", "A07:2021 Identification and Authentication Failures (Brute Force / Credential Stuffing)"),
    }
    
    sections = []
    for filename, (tech_id, name) in owasp_mapping.items():
        doc_path = owasp_dir / filename
        if not doc_path.exists():
            continue
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        sections.append({
            "source": "OWASP Top 10 (2021)",
            "technique_id": tech_id,
            "technique_name": name,
            "tactics": "Web Application Security",
            "section_type": "web_vulnerability",
            "title": f"OWASP Top 10: {name}",
            "text": f"OWASP Top 10 (2021 Edition): {name}\n\n{content}"
        })
        
    print(f"[Parser] Extracted {len(sections)} security risk profiles from OWASP Top 10.")
    return sections

def build_chunk_corpus(grid_id: str, chunk_size: int, overlap: int, parsed_sections: list[dict]) -> list[dict]:
    chunks = []
    chunk_counter = 0
    for sec in parsed_sections:
        raw_text = sec["text"]
        sub_chunks = sliding_window_chunk(raw_text, chunk_size, overlap)
        for i, text_chunk in enumerate(sub_chunks):
            cid = f"{grid_id}_{sec['technique_id']}_{sec['section_type']}_{i}"
            # Ensure unique clean ID
            cid = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', cid)
            chunks.append({
                "chunk_id": cid,
                "grid_id": grid_id,
                "source": sec["source"],
                "technique_id": sec["technique_id"],
                "technique_name": sec["technique_name"],
                "tactics": sec["tactics"],
                "section_type": sec["section_type"],
                "title": sec["title"],
                "chunk_index": i,
                "total_chunks_in_section": len(sub_chunks),
                "word_count": len(text_chunk.split()),
                "text": text_chunk
            })
            chunk_counter += 1
    return chunks

def main():
    raw_dir = Path("artifacts/rag/raw")
    out_dir = Path("artifacts/rag")
    
    # 1. Parse all authoritative primary snapshots
    mitre_sections = parse_mitre_stix(raw_dir / "enterprise-attack-20.0.json")
    nist_sections = parse_nist_pdf(raw_dir / "NIST.SP.800-61r3.pdf")
    nvd_sections = parse_nvd_json(raw_dir / "nvd_cve_2014_0160.json")
    owasp_sections = parse_owasp_docs(raw_dir / "owasp_top10_2021")
    
    all_sections = mitre_sections + nist_sections + nvd_sections + owasp_sections
    print(f"\n[Parser] Total parsed authoritative document sections: {len(all_sections)}")
    
    # 2. Generate chunks for each pre-registered grid
    grid_configs = [
        ("C256", 256, 50),
        ("C512", 512, 100),
        ("C1024", 1024, 200),
    ]
    
    for grid_id, chunk_size, overlap in grid_configs:
        t0 = time.time()
        chunks = build_chunk_corpus(grid_id, chunk_size, overlap, all_sections)
        t_el = time.time() - t0
        
        out_file = out_dir / f"chunks_{grid_id.lower()}_v1.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "grid_id": grid_id,
                "chunk_size_words": chunk_size,
                "overlap_words": overlap,
                "total_chunks": len(chunks),
                "generation_time_s": round(t_el, 3),
                "chunks": chunks
            }, f, indent=2)
            
        print(f"[Chunker] {grid_id} (size={chunk_size}, overlap={overlap}): generated {len(chunks)} chunks -> {out_file} ({t_el:.2f}s)")
        
    print("\n[Parser] All chunk grids generated successfully.")

if __name__ == "__main__":
    main()

