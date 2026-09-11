"""Run-scoped, atomic artifact persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class ArtifactSecurityError(ValueError):
    pass


class ArtifactWorkspace:
    # Broad evaluation runs persist complete search/fetch traces for auditability.
    # Keep hard bounds, but size them for the 200-query evaluation profile.
    MAX_FILE_BYTES = 32 * 1024 * 1024
    MAX_RUN_BYTES = 128 * 1024 * 1024

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_id):
            raise ArtifactSecurityError("run_id contains unsafe characters")
        path = (self.root / run_id).resolve()
        if path.parent != self.root:
            raise ArtifactSecurityError("run directory escaped artifact root")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve(self, run_id: str, relative_path: str) -> Path:
        if Path(relative_path).is_absolute():
            raise ArtifactSecurityError("absolute artifact paths are forbidden")
        run_dir = self.run_dir(run_id)
        target = (run_dir / relative_path).resolve()
        if run_dir not in target.parents:
            raise ArtifactSecurityError("artifact path escaped run directory")
        current = run_dir
        for part in Path(relative_path).parts[:-1]:
            current = current / part
            if current.exists() and current.is_symlink():
                raise ArtifactSecurityError("symlink directories are forbidden")
        return target

    def _write_bytes(self, run_id: str, relative_path: str, data: bytes) -> Path:
        if len(data) > self.MAX_FILE_BYTES:
            raise ArtifactSecurityError("artifact exceeds per-file size limit")
        target = self.resolve(run_id, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existing_total = sum(
            item.stat().st_size for item in self.run_dir(run_id).rglob("*") if item.is_file()
        )
        if existing_total + len(data) > self.MAX_RUN_BYTES:
            raise ArtifactSecurityError("run artifacts exceed total size limit")
        fd, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return target

    def write_json(self, run_id: str, relative_path: str, value: Any) -> Path:
        data = json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        return self._write_bytes(run_id, relative_path, data)

    def write_text(self, run_id: str, relative_path: str, value: str) -> Path:
        return self._write_bytes(run_id, relative_path, value.encode("utf-8"))

    def write_jsonl(self, run_id: str, relative_path: str, values: list[Any]) -> Path:
        data = "".join(
            json.dumps(value, ensure_ascii=False, default=str) + "\n" for value in values
        )
        return self.write_text(run_id, relative_path, data)

    @staticmethod
    def file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _audit_log(state: dict[str, Any]) -> dict[str, Any]:
        contributions = state.get("contributions", [])
        latest: dict[str, dict[str, Any]] = {}
        for item in sorted(contributions, key=lambda value: value.get("attempt", 0)):
            latest[item.get("task_id", "")] = item
        # Full result candidates remain in details/traces/search.json. Repeating
        # every snippet in the aggregate audit log makes broad fanout runs grow
        # quadratically and can exceed the artifact safety limit.
        search_summaries = []
        for item in state.get("search_traces", []):
            summary = {key: value for key, value in item.items() if key != "results"}
            summary["result_count"] = len(item.get("results", []))
            search_summaries.append(summary)
        result = {
            "run": {
                "run_id": state["run_id"],
                "status": state.get("status"),
                "mode": state.get("request", {}).get("mode"),
                "budget": state.get("run_budget", {}),
                "round_count": state.get("round_no", 0),
                "synthesis_decision": state.get("synthesis_decision", {}),
            },
            "workflow_events": state.get("events", []),
            "plans": state.get("plan_history", []),
            "tasks": state.get("tasks", []),
            "task_attempts": contributions,
            "task_sufficiency": list(latest.values()),
            "llm_outputs": {
                "planner": state.get("planner_traces", []),
                "worker": state.get("worker_traces", []),
                "agent_loop": state.get("agent_loop_traces", []),
                "extraction": state.get("extraction_traces", []),
                "synthesizer": state.get("synthesis_traces", []),
                "report": state.get("writer_trace", {}),
            },
            "tool_outputs": {
                "search": search_summaries,
                "fetch": state.get("fetch_traces", []),
            },
            "ledger": {
                "sources": state.get("sources", []),
                "evidence": state.get("evidence", []),
                "claims": state.get("claims", []),
                "claim_evidence_links": state.get("links", []),
                "citations": state.get("citations", []),
                "source_leads": state.get("source_leads", []),
                "entities": state.get("entity_candidates", []),
                "evidence_bindings": state.get("evidence_bindings", []),
                "evidence_edges": state.get("evidence_edges", []),
            },
            "search_state": {
                "schemas": state.get("search_schemas", []),
                "schema_revisions": state.get("schema_revisions", []),
                "cell_coverage": state.get("cell_coverage", {}),
                "scope_completeness": state.get("scope_completeness", {}),
                "frontier_history": state.get("frontier_history", []),
                "sensor_history": state.get("sensor_history", []),
                "failure_memory": state.get("failure_memory", []),
                "observation_commits": state.get("observation_commits", []),
                "observation_rejections": state.get("observation_rejections", []),
            },
            "usage_events": state.get("usage_events", []),
            "citation_gate": state.get("gate_result", {}),
        }
        if state.get("skill_evolution_observation"):
            result["skill_evolution"] = {
                "observation": state.get("skill_evolution_observation", {}),
                "signals": state.get("evolution_signals", []),
                "episodes": state.get("evolution_episodes", []),
            }
        return result

    @staticmethod
    def _requirement_matrix(state: dict[str, Any]) -> dict[str, Any]:
        """Materialize only planner-visible requirements, never hidden rubrics."""
        requirements = list(
            {
                str(item["requirement_id"]): item
                for plan in state.get("plan_history", [])
                for item in plan.get("requirements", [])
                if isinstance(item, dict)
                and item.get("requirement_id")
                and item.get("text")
            }.values()
        )
        if not requirements:
            requirements = [
                {
                    "requirement_id": f"req-{index:03d}",
                    "text": text,
                    "kind": "fact",
                }
                for index, text in enumerate(
                    dict.fromkeys(
                        str(item).strip()
                        for plan in state.get("plan_history", [])
                        for item in plan.get("answer_requirements", [])
                        if str(item).strip()
                    ),
                    1,
                )
            ]
        coverage = {
            str(item.get("requirement_id")): item
            for item in state.get("synthesis_decision", {}).get("coverage", [])
            if item.get("requirement_id")
        }
        tasks = state.get("tasks", [])
        searches = state.get("search_traces", [])
        fetches = state.get("fetch_traces", [])
        evidence = state.get("evidence", [])
        claims = state.get("claims", [])
        output = state.get("structured_output", {})

        def matching_ids(items: list[dict[str, Any]], requirement_id: str, field: str) -> list[str]:
            return sorted(
                {
                    str(item.get(field))
                    for item in items
                    if requirement_id in item.get("requirement_ids", []) and item.get(field)
                }
            )

        placeholder_markers = (
            "not verified",
            "writer was unavailable",
            "writer timed out",
            "did not complete",
            "未核实",
            "writer 不可用",
            "writer 超时",
            "没有分配到可发布",
        )

        def normalized(value: Any) -> str:
            return " ".join(str(value or "").casefold().split())

        def publishable_text(value: Any, *, cell_type: str = "") -> bool:
            text = normalized(value)
            return bool(text) and cell_type.casefold() != "uncertainty" and not any(
                marker in text for marker in placeholder_markers
            )

        def section_matches(requirement: dict[str, Any], section: dict[str, Any]) -> bool:
            target = normalized(requirement.get("target_section"))
            title = normalized(section.get("title"))
            if target:
                return bool(title) and (target in title or title in target)
            if str(requirement.get("kind") or "fact") in {"table", "structure"}:
                text = normalized(requirement.get("text"))
                return bool(title) and title in text
            return True

        matrix_rows: list[dict[str, Any]] = []
        for requirement in requirements:
            requirement_id = str(requirement["requirement_id"])
            kind = str(requirement.get("kind") or "fact").casefold()
            claim_ids = matching_ids(claims, requirement_id, "claim_id")
            evidence_ids = matching_ids(evidence, requirement_id, "evidence_id")
            task_ids = matching_ids(tasks, requirement_id, "task_id")
            query_ids = matching_ids(searches, requirement_id, "query_id")
            fetch_trace_ids = matching_ids(fetches, requirement_id, "trace_id")

            attached_paragraphs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            attached_tables: list[tuple[dict[str, Any], dict[str, Any]]] = []
            attached_cells: list[tuple[dict[str, Any], dict[str, Any]]] = []
            matching_sections = [
                section
                for section in output.get("sections", [])
                if section_matches(requirement, section)
            ]
            for section in output.get("sections", []):
                for paragraph in section.get("paragraphs", []):
                    if requirement_id in paragraph.get("requirement_ids", []):
                        attached_paragraphs.append((section, paragraph))
                for table in section.get("tables", []):
                    table_has_requirement = requirement_id in table.get(
                        "requirement_ids", []
                    )
                    if table_has_requirement:
                        attached_tables.append((section, table))
                    for row in table.get("rows", []):
                        for cell in row:
                            if requirement_id in cell.get("requirement_ids", []):
                                attached_cells.append((section, cell))
                                if not table_has_requirement:
                                    attached_tables.append((section, table))

            semantic_paragraphs = [
                (section, paragraph)
                for section, paragraph in attached_paragraphs
                if publishable_text(
                    paragraph.get("text"),
                    cell_type=str(paragraph.get("paragraph_type") or ""),
                )
                and section_matches(requirement, section)
                and (
                    kind == "analysis"
                    or bool(set(paragraph.get("claim_ids", [])) & set(claim_ids))
                )
            ]
            semantic_cells = [
                (section, cell)
                for section, cell in attached_cells
                if publishable_text(
                    cell.get("text"), cell_type=str(cell.get("cell_type") or "")
                )
                and section_matches(requirement, section)
                and bool(set(cell.get("claim_ids", [])) & set(claim_ids))
            ]
            semantic_tables = [
                (section, table)
                for section, table in attached_tables
                if section_matches(requirement, section)
                and any(
                    publishable_text(
                        cell.get("text"),
                        cell_type=str(cell.get("cell_type") or ""),
                    )
                    for row in table.get("rows", [])
                    for cell in row
                )
                and any(
                    bool(set(cell.get("claim_ids", [])) & set(claim_ids))
                    for row in table.get("rows", [])
                    for cell in row
                )
            ]

            structure_sections_with_content = [
                section
                for section in matching_sections
                if any(
                    publishable_text(
                        paragraph.get("text"),
                        cell_type=str(paragraph.get("paragraph_type") or ""),
                    )
                    for paragraph in section.get("paragraphs", [])
                )
                or any(
                    publishable_text(
                        cell.get("text"),
                        cell_type=str(cell.get("cell_type") or ""),
                    )
                    for table in section.get("tables", [])
                    for row in table.get("rows", [])
                    for cell in row
                )
            ]

            report_text = " ".join(
                [paragraph.get("text", "") for _, paragraph in semantic_paragraphs]
                + [
                    " ".join(
                        [table.get("title", ""), *table.get("columns", [])]
                        + [
                            str(cell.get("text") or "")
                            for row in table.get("rows", [])
                            for cell in row
                        ]
                    )
                    for _, table in semantic_tables
                ]
            )
            if kind == "structure":
                report_text += " " + " ".join(
                    " ".join(
                        [
                            str(section.get("title") or ""),
                            *[
                                str(paragraph.get("text") or "")
                                for paragraph in section.get("paragraphs", [])
                                if publishable_text(
                                    paragraph.get("text"),
                                    cell_type=str(
                                        paragraph.get("paragraph_type") or ""
                                    ),
                                )
                            ],
                        ]
                    )
                    for section in structure_sections_with_content
                )
            required_elements = [
                normalized(item)
                for item in requirement.get("required_elements", [])
                if normalized(item)
            ]
            missing_elements = [
                item for item in required_elements if item not in normalized(report_text)
            ]
            correct_shape = (
                bool(semantic_tables)
                if kind == "table"
                else bool(structure_sections_with_content)
                if kind == "structure"
                else bool(semantic_paragraphs or semantic_cells)
            )
            placement_ok = (
                bool(structure_sections_with_content)
                if kind == "structure"
                else bool(semantic_paragraphs or semantic_cells or semantic_tables)
            )
            semantically_complete = correct_shape and placement_ok and not missing_elements

            has_partial_material = bool(
                attached_paragraphs
                or attached_cells
                or attached_tables
                or claim_ids
                or evidence_ids
                or (kind in {"table", "structure"} and matching_sections)
            )
            status = (
                "reported"
                if semantically_complete
                else "partial"
                if has_partial_material
                else "searched"
                if query_ids or task_ids or fetch_trace_ids
                else "missing"
            )
            matrix_rows.append(
                {
                    **requirement,
                    "task_ids": task_ids,
                    "query_ids": query_ids,
                    "fetch_trace_ids": fetch_trace_ids,
                    "evidence_ids": evidence_ids,
                    "claim_ids": claim_ids,
                    "report_paragraph_ids": sorted(
                        paragraph.get("paragraph_id")
                        for _, paragraph in attached_paragraphs
                        if paragraph.get("paragraph_id")
                    ),
                    "table_cell_ids": sorted(
                        cell.get("cell_id")
                        for _, cell in attached_cells
                        if cell.get("cell_id")
                    ),
                    "status": status,
                    "semantic_checks": {
                        "correct_shape": correct_shape,
                        "correct_section_placement": placement_ok,
                        "non_placeholder_content": bool(
                            semantic_paragraphs
                            or semantic_cells
                            or semantic_tables
                            or structure_sections_with_content
                        ),
                        "missing_required_elements": missing_elements,
                    },
                    "coverage": coverage.get(requirement_id, {}),
                }
            )

        return {
            "source": "planner-visible question and research plan only",
            "hidden_rubrics_visible": False,
            "requirements": matrix_rows,
        }

    @staticmethod
    def _audit_report(state: dict[str, Any]) -> str:
        contributions = state.get("contributions", [])
        latest: dict[str, dict[str, Any]] = {}
        for item in sorted(contributions, key=lambda value: value.get("attempt", 0)):
            latest[item.get("task_id", "")] = item
        task_map = {item.get("task_id"): item for item in state.get("tasks", [])}
        searches = state.get("search_traces", [])
        fetches = state.get("fetch_traces", [])
        extractions = state.get("extraction_traces", [])
        planners = state.get("planner_traces", [])
        workers = state.get("worker_traces", [])
        syntheses = state.get("synthesis_traces", [])
        report = state.get("writer_trace", {})
        fetch_failed = sum(item.get("status") == "failed" for item in fetches)
        fetch_empty = sum(item.get("status") == "no_evidence" for item in fetches)
        fetch_success = sum(item.get("status") == "fetched" for item in fetches)
        extraction_calls = sum(
            bool(item.get("input_tokens", 0) or item.get("output_tokens", 0))
            for item in extractions
        )
        search_results = sum(len(item.get("results", [])) for item in searches)
        quality_threshold = min(
            (
                item.get("sufficiency", {}).get("max_source_quality_tier", 3)
                for item in state.get("tasks", [])
            ),
            default=3,
        )
        effective_source_uris = {
            item.get("canonical_uri")
            for item in state.get("sources", [])
            if item.get("quality_tier", 4) <= quality_threshold
            and item.get("canonical_uri")
        }
        lines = [
            f"# 研究审计报告：{state['run_id']}",
            "",
            f"- 运行状态：`{state.get('status', 'unknown')}`",
            f"- 研究轮次：{state.get('round_no', 0)}",
            f"- 最终综合决策：`{state.get('synthesis_decision', {}).get('decision', 'unknown')}`",
            f"- Search 调用：{len(searches)}",
            f"- Search 候选结果：{search_results}",
            (
                f"- Fetch 尝试：{len(fetches)}（成功抽取 {fetch_success}，"
                f"无有效证据 {fetch_empty}，失败 {fetch_failed}）"
            ),
            f"- 成功抽取：{sum(item.get('successful_extractions', 0) for item in latest.values())}",
            (
                f"- 来源记录：{len(state.get('sources', []))}，"
                f"有效唯一 URL：{len(effective_source_uris)}"
            ),
            f"- Ledger Evidence：{len(state.get('evidence', []))}",
            f"- Claim：{len(state.get('claims', []))}",
            (
                "- LLM 调用："
                f"Planner {sum(bool(item.get('input_tokens') or item.get('output_tokens')) for item in planners)}，"
                f"Worker {sum(bool(item.get('input_tokens') or item.get('output_tokens')) for item in workers)}，"
                f"Extraction {extraction_calls}，"
                f"Synthesizer {sum(bool(item.get('input_tokens') or item.get('output_tokens')) for item in syntheses)}，"
                f"Report {int(bool(report.get('input_tokens') or report.get('output_tokens')))}"
            ),
            "",
            "## 任务充分性",
            "",
            "| 研究维度 | 轮次 | 查询 | 查询覆盖 | 成功抽取 | 有效来源 | 独立域名 | 有效证据 | 质量≤ | 达标 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for task_id, item in latest.items():
            task = task_map.get(task_id, {})
            floor = task.get("sufficiency", {})
            dimension = str(task.get("research_dimension", task_id)).replace("|", "\\|")
            lines.append(
                "| "
                + " | ".join(
                    [
                        dimension,
                        str(item.get("attempt", 0)),
                        f"{item.get('query_count', 0)}/{floor.get('min_queries', 1)}",
                        f"{len(item.get('covered_query_ids', []))}/{floor.get('min_query_coverage', 1)}",
                        f"{item.get('successful_extractions', 0)}/{floor.get('min_successful_extractions', 1)}",
                        f"{item.get('effective_source_count', 0)}/{floor.get('min_effective_sources', 1)}",
                        f"{item.get('independent_domain_count', 0)}/{floor.get('min_independent_domains', 1)}",
                        f"{item.get('evidence_count', 0)}/{floor.get('min_evidence', 1)}",
                        str(floor.get("max_source_quality_tier", 3)),
                        "是" if item.get("sufficiency_passed") else "否",
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "## 任务尝试",
                "",
                "| 研究维度 | 轮次 | 状态 | 停止原因 | 成功抽取 | 有效证据 | 达标 |",
                "| --- | ---: | --- | --- | ---: | ---: | --- |",
            ]
        )
        for item in contributions:
            task = task_map.get(item.get("task_id"), {})
            dimension = str(
                task.get("research_dimension", item.get("task_id", ""))
            ).replace("|", "\\|")
            lines.append(
                "| "
                + " | ".join(
                    [
                        dimension,
                        str(item.get("attempt", 0)),
                        str(item.get("status", "")),
                        str(item.get("stop_reason", "")).replace("|", "\\|"),
                        str(item.get("successful_extractions", 0)),
                        str(item.get("evidence_count", 0)),
                        "是" if item.get("sufficiency_passed") else "否",
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "## 来源质量",
                "",
                "| Tier | 来源 | 域名 | 质量说明 |",
                "| ---: | --- | --- | --- |",
            ]
        )
        for source in state.get("sources", []):
            uri = str(source.get("canonical_uri", ""))
            domain = str(source.get("provenance", {}).get("independent_domain", ""))
            title = str(source.get("title", "")).replace("|", "\\|")
            reason = str(source.get("quality_reason", "")).replace("|", "\\|")
            lines.append(
                f"| {source.get('quality_tier', '')} | [{title}]({uri}) | {domain} | {reason} |"
            )
        lines.extend(
            [
                "",
                "## 流程事件",
                "",
                "| 序号 | 事件 | Task | 状态 |",
                "| ---: | --- | --- | --- |",
            ]
        )
        for event in state.get("events", []):
            lines.append(
                f"| {event.get('sequence', '')} | {event.get('event_type', '')} | "
                f"{event.get('task_id') or ''} | {event.get('status') or ''} |"
            )
        lines.extend(
            [
                "",
                "## 审计文件",
                "",
                "- [完整结构化审计日志](audit-log.json)",
                "- [工作流事件](../runtime/events.jsonl)",
                "- [Planner 各轮输出](../traces/planner.json)",
                "- [Worker 策略与反思](../traces/worker.json)",
                "- [Worker ReAct Skill Loop](../traces/agent-loop.json)",
                "- [Worker 候选 URL 选择](../traces/candidate-selection.json)",
                "- [Synthesizer 各轮判断](../traces/synthesis.json)",
                "- [Search 结果](../traces/search.json)",
                "- [Fetch 结果](../traces/fetch.json)",
                "- [抽取输出](../traces/extraction.json)",
                "- [最终报告综合输出](../traces/report.json)",
                "",
            ]
        )
        return "\n".join(lines)

    def persist_run(self, state: dict[str, Any]) -> tuple[str, str]:
        run_id = state["run_id"]
        output_name = "report.md"
        files = {
            "details/input/request.json": state["request"],
            "details/runtime/p0-ablation.json": state.get("p0_ablation", {}),
            "details/planning/research-plan.json": {
                "scope": state["scope"],
                "plans": state.get("plan_history", []),
                "tasks": state.get("tasks", []),
            },
            "details/graph/dag-validation.json": state.get("dag_validation", {}),
            "details/graph/task-runtime.json": state.get("task_runtime", []),
            "details/graph/wave-history.json": state.get("wave_history", []),
            "details/graph/requirement-status.json": state.get("requirement_status", []),
            "details/graph/candidate-leads.json": state.get("candidate_leads", []),
            "details/search/search-schemas.json": state.get("search_schemas", []),
            "details/search/schema-revisions.json": state.get("schema_revisions", []),
            "details/search/source-leads.json": state.get("source_leads", []),
            "details/search/entities.json": state.get("entity_candidates", []),
            "details/search/citation-matrix.json": state.get("evidence_bindings", []),
            "details/search/evidence-graph.json": state.get("evidence_edges", []),
            "details/search/cell-coverage.json": state.get("cell_coverage", {}),
            "details/search/scope-completeness.json": state.get("scope_completeness", {}),
            "details/search/frontier-history.json": state.get("frontier_history", []),
            "details/search/sensor-history.json": state.get("sensor_history", []),
            "details/search/failure-memory.json": state.get("failure_memory", []),
            "details/search/observation-commits.json": state.get("observation_commits", []),
            "details/search/observation-rejections.json": state.get("observation_rejections", []),
            "details/evidence/sources.json": state.get("sources", []),
            "details/evidence/source-snapshots.json": state.get("source_snapshots", []),
            "details/evidence/evidence.json": state.get("evidence", []),
            "details/evidence/claims.json": state.get("claims", []),
            "details/evidence/claim-evidence-links.json": state.get("links", []),
            "details/evidence/citations.json": state.get("citations", []),
            "details/evidence/citation-assessment.json": state.get("gate_result", {"passed": False}),
            "details/traces/planner.json": state.get("planner_traces", []),
            "details/traces/worker.json": state.get("worker_traces", []),
            "details/traces/agent-loop.json": state.get("agent_loop_traces", []),
            "details/traces/candidate-selection.json": [
                item
                for item in state.get("agent_loop_traces", [])
                if item.get("phase") == "candidate_selection"
            ],
            "details/traces/synthesis.json": state.get("synthesis_traces", []),
            "details/traces/search.json": state.get("search_traces", []),
            "details/traces/fetch.json": state.get("fetch_traces", []),
            "details/traces/extraction.json": state.get("extraction_traces", []),
            "details/traces/report.json": state.get("writer_trace", {}),
            "details/output/structured-output.json": state.get("structured_output", {}),
            "details/runtime/usage.json": state.get("usage_events", []),
            "details/audit/audit-log.json": self._audit_log(state),
        }
        if state.get("skill_evolution_observation"):
            files.update(
                {
                    "details/skill-evolution/observation.json": state.get(
                        "skill_evolution_observation", {}
                    ),
                    "details/skill-evolution/signals.json": state.get(
                        "evolution_signals", []
                    ),
                    "details/skill-evolution/episodes.json": state.get(
                        "evolution_episodes", []
                    ),
                }
            )
        if state.get("request", {}).get("profile_id") == "drb_evaluation":
            files["details/evaluation/requirement-matrix.json"] = (
                self._requirement_matrix(state)
            )
        written: dict[str, str] = {}
        for name, value in files.items():
            path = self.write_json(run_id, name, value)
            written[name] = self.file_hash(path)
        events_name = "details/runtime/events.jsonl"
        events_path = self.write_jsonl(run_id, events_name, state.get("events", []))
        written[events_name] = self.file_hash(events_path)
        controller_name = "details/graph/controller-decisions.jsonl"
        controller_path = self.write_jsonl(
            run_id, controller_name, state.get("controller_history", [])
        )
        written[controller_name] = self.file_hash(controller_path)
        projection_name = "details/graph/context-projections.jsonl"
        projection_path = self.write_jsonl(
            run_id, projection_name, state.get("context_projection_traces", [])
        )
        written[projection_name] = self.file_hash(projection_path)
        audit_name = "details/audit/audit-report.md"
        audit_report_path = self.write_text(run_id, audit_name, self._audit_report(state))
        written[audit_name] = self.file_hash(audit_report_path)
        output_path = self.write_text(run_id, output_name, state.get("markdown", ""))
        written[output_name] = self.file_hash(output_path)
        manifest = {
            "run_id": run_id,
            "schema_version": "3.0",
            "mode": state["request"]["mode"],
            "profile_id": state["request"]["profile_id"],
            "status": state["status"],
            "budget": state.get("run_budget", {}),
            "artifact_hashes": written,
        }
        self.write_json(run_id, "details/manifest.json", manifest)
        return str(self.run_dir(run_id)), str(output_path)
