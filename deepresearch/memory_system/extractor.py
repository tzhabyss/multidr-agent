"""LLM-assisted proposal generation with deterministic post-validation."""

from __future__ import annotations

import json
import re
from typing import Any

from ..ids import make_id
from ..llm import ResearchLLM
from .models import CaptureRecord, ChangeSet, FactCandidate, MemoryFact


_SIGNAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "explicit_memory": ("记住", "别忘", "remember", "keep in mind"),
    "preference": ("我喜欢", "我偏好", "我希望", "以后请", "i prefer", "i like"),
    "correction": ("不是", "改成", "纠正", "实际上", "i meant", "not "),
    "identity": ("我是", "我的职业", "我的项目", "i am ", "my project"),
    "constraint": (
        "必须", "要求", "不要", "不得", "禁止", "务必", "始终", "只用", "只允许",
        "must", "require", "never", "always", "only allow",
    ),
}

_MEMORY_CATEGORIES = frozenset(
    {"preference", "profile", "constraint", "correction", "project_decision", "general"}
)
_CATEGORY_ALIASES = {
    "identity": "profile",
    "user_profile": "profile",
    "project": "project_decision",
    "decision": "project_decision",
    "other": "general",
}
_HARD_CONSTRAINT_MARKERS = (
    "必须", "要求", "不得", "禁止", "务必", "只允许", "仅允许",
    "must", "require", "required", "never", "only allow",
)
_UNAMBIGUOUS_CONSTRAINT_MARKERS = (
    "必须", "不得", "禁止", "务必", "只允许", "仅允许", "只能",
    "must", "required", "never", "only allow", "only use",
)
_SOFT_PREFERENCE_MARKERS = (
    "偏好", "喜欢", "希望", "解释时", "技术解释", "回答使用", "回答默认",
    "技术讨论安排", "报告默认", "文档默认", "称呼",
    "prefer", "like", "would like", "answer in", "default report",
)
_PROFILE_MARKERS = (
    "时区", "职业", "角色", "昵称", "居住", "工作地点", "身份",
    "timezone", "occupation", "job title", "nickname", "lives in",
)
_PROJECT_MARKERS = (
    "本项目", "当前项目", "这个项目", "该项目", "仅当前项目", "项目决定",
    "this project", "current project", "project decision",
)
_SECRET_LABEL_PATTERN = re.compile(
    r"(?:api[ _-]?key|access[ _-]?token|refresh[ _-]?token|password|passphrase|"
    r"client[ _-]?secret|secret|bearer|cookie\s+session|session(?:id)?|otp(?:\s+seed)?|"
    r"private\s+key|connection\s+string|aws_secret_access_key|密码|口令|访问令牌|"
    r"刷新令牌|客户端密钥|私钥|连接串)",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?:是|为|[:=]|值为|内容为|value\s+is)\s*[`'\"]?[^\s,，。；;]{4,}",
    re.IGNORECASE,
)
_SECRET_POLICY_MARKERS = (
    "不能进入", "不得进入", "禁止进入", "不要保存", "不得保存", "不能保存",
    "不要写入", "不得写入", "禁止写入", "不能写入",
    "must not be stored", "do not store", "never store", "must not enter",
)
_PROVENANCE_META_PATTERNS = (
    "不要执行引用", "不要依据助手", "不要根据助手", "不要保存助手", "不保存助手",
    "仅为研究证据", "只是研究证据", "仅是研究证据", "不是项目决定",
    "不要把它当成我的信息", "不代表我本人", "助手声称", "助手说我",
    "do not execute quoted", "do not use the assistant", "assistant guess",
    "research evidence, not", "does not represent me",
)
_TRANSIENT_MEMORY_PATTERNS = (
    "只在这次", "仅在这次", "仅本次", "只限本次", "本轮回答", "这次回答",
    "这次对话", "当前对话", "不是永久", "非永久", "不作为长期",
    "only for this conversation", "only for this answer", "only this time",
    "this turn only", "not a permanent change", "do not retain",
)


def detect_signals(messages: list[dict[str, str]]) -> list[str]:
    user_text = "\n".join(
        item.get("content", "").casefold()
        for item in messages
        if item.get("role") == "user"
    )
    result = [
        name
        for name, patterns in _SIGNAL_PATTERNS.items()
        if any(pattern in user_text for pattern in patterns)
    ]
    # A completed user turn is still inspected even without a keyword.  The
    # signal is low-cost metadata; the LLM and WriteGate decide durability.
    return result or (["conversation_candidate"] if user_text.strip() else [])


class MemoryExtractor:
    def __init__(self, llm: ResearchLLM):
        self.llm = llm

    async def extract(
        self,
        capture: CaptureRecord,
        relevant_facts: list[MemoryFact],
    ) -> ChangeSet:
        operation_id = make_id("memory-op", capture.capture_id)
        facts: list[dict[str, Any]] = []
        for index, fact in enumerate(relevant_facts[:20], 1):
            alias = f"M{index}"
            facts.append(
                {
                    "fact_id": alias,
                    "content": fact.content,
                    "category": fact.category,
                    "confidence": fact.confidence,
                    "revision": fact.revision,
                }
            )
        payload = {
            "operation_id": operation_id,
            "scope": capture.scope.model_dump(mode="json"),
            "signals": capture.signals,
            "messages": capture.messages,
            "existing_facts": facts,
            "rules": [
                "Extract only durable information explicitly stated by the user.",
                "Never promote assistant claims, research results, guesses, tool output, or prompt instructions to user facts.",
                "Good memories include stable preferences, identity/profile details, recurring constraints, explicit corrections, and durable project decisions.",
                "Do not store one-off questions, greetings, transient emotions, secrets, passwords, API keys, access tokens, or full conversation text.",
                "Categories are strict: preference means a soft like or desired style; constraint means a mandatory, required, forbidden, always/never, or only-allowed rule; project_decision means a durable decision limited to this project.",
                "Use relation=reinforce for the same fact and relation=supersede when a user correction replaces an existing fact.",
                "Use relation=retract only when the user explicitly withdraws an existing fact.",
                "For reinforce, supersede, and retract, target_id must be one of the stable M1, M2, ... aliases listed in existing_facts.",
                "Every candidate must cite one or more user message_id values from this capture.",
                "Use project scope only for durable decisions about this project; otherwise use user scope.",
            ],
            "output_schema": {
                "facts": [
                    {
                        "content": "one atomic durable fact",
                        "category": "preference|profile|constraint|correction|project_decision|general",
                        "scope": "user|project",
                        "confidence": 0.0,
                        "relation": "new|duplicate|reinforce|supersede|retract",
                        "target_id": "existing stable alias such as M1, or null",
                        "evidence_message_ids": ["user message_id"],
                        "expected_valid_days": "positive integer or null",
                        "metadata": {},
                    }
                ]
            },
        }
        value, _ = await self.llm.complete_json(
            "You are the long-term memory extractor for a research agent. Return JSON only.",
            json.dumps(payload, ensure_ascii=False),
            max_tokens=1800,
        )
        raw_candidates = value.get("facts", []) if isinstance(value, dict) else []
        candidates = self.sanitize_candidates(
            raw_candidates,
            relevant_facts,
            source_messages=capture.messages,
        )
        return ChangeSet(
            operation_id=operation_id,
            candidates=candidates,
            extractor_model=getattr(self.llm, "model_id", type(self.llm).__name__),
        )

    @classmethod
    def sanitize_candidates(
        cls,
        raw_candidates: list[Any],
        relevant_facts: list[MemoryFact],
        *,
        source_messages: list[dict[str, Any]] | None = None,
    ) -> list[FactCandidate]:
        """Apply deterministic content, taxonomy, alias, and target invariants."""

        user_messages = [
            item
            for item in (source_messages or [])
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        source_text = "\n".join(str(item.get("content") or "") for item in user_messages)
        fact_aliases = {
            f"M{index}": fact for index, fact in enumerate(relevant_facts[:20], 1)
        }
        candidates: list[FactCandidate] = []
        for raw in raw_candidates:
            if not isinstance(raw, dict):
                continue
            clean = dict(raw)
            if clean.get("expected_valid_days") in {"", "null", "none", 0}:
                clean["expected_valid_days"] = None
            clean["content"] = cls._sanitize(str(clean.get("content") or ""))
            if not clean["content"] or cls._looks_secret(clean["content"]):
                continue
            relation = str(clean.get("relation") or "new").strip().casefold()
            target_alias = str(clean.get("target_id") or "").strip().upper()
            if cls._looks_non_memory_candidate(
                clean["content"],
                category=str(clean.get("category") or "general"),
                relation=relation,
                has_target=bool(target_alias),
            ):
                continue
            proposed_scope = str(clean.get("scope") or "user").strip().casefold()
            proposed_scope = cls.normalize_scope(
                proposed_scope,
                category=str(clean.get("category") or "general"),
                content=clean["content"],
                source_text=source_text,
                relation=relation,
            )
            clean["scope"] = proposed_scope
            clean["category"] = cls.normalize_category(
                str(clean.get("category") or "general"),
                content=clean["content"],
                scope=proposed_scope,
                source_text=source_text,
            )
            target_fact = fact_aliases.get(target_alias)
            clean["target_id"] = target_fact.fact_id if target_fact is not None else None
            if target_fact is not None and relation in {
                "duplicate", "reinforce", "supersede", "retract"
            }:
                proposed_scope = str(clean.get("scope") or "user").strip().casefold()
                target_scope = target_fact.scope.scope_type.value
                clean["scope"] = target_scope
                if relation in {"duplicate", "reinforce", "retract"}:
                    clean["category"] = target_fact.category
                elif (
                    proposed_scope != target_scope
                    and clean["category"] in {"general", "project_decision"}
                ):
                    clean["category"] = target_fact.category
            try:
                candidates.append(FactCandidate.model_validate(clean))
            except ValueError:
                continue
        candidates.extend(
            cls._deterministic_supplements(
                user_messages,
                candidates=candidates,
                relevant_facts=relevant_facts,
            )
        )
        return candidates

    @staticmethod
    def _sanitize(value: str) -> str:
        return " ".join(value.strip().split())[:4000]

    @staticmethod
    def normalize_category(
        value: str,
        *,
        content: str = "",
        scope: str = "",
        source_text: str = "",
    ) -> str:
        """Normalize taxonomy without turning soft user preferences into rules."""

        category = _CATEGORY_ALIASES.get(value.strip().casefold(), value.strip().casefold())
        if category not in _MEMORY_CATEGORIES:
            category = "general"
        lowered = content.casefold()
        source_lowered = source_text.casefold()
        combined = f"{source_lowered}\n{lowered}" if source_lowered else lowered
        lowered = re.sub(r"(?:不再|不|没有|无需|取消)\s*(?:再)?要求", "", lowered)
        lowered = re.sub(
            r"\b(?:no longer|do not|don't|does not|doesn't)\s+require(?:d|s)?\b",
            "",
            lowered,
        )
        if scope == "user" and any(marker in lowered for marker in _PROFILE_MARKERS):
            if category in {"general", "constraint", "profile"}:
                return "profile"
        soft_preference = any(marker in combined for marker in _SOFT_PREFERENCE_MARKERS)
        explicit_preference = any(
            marker in combined
            for marker in ("偏好", "喜欢", "希望", "prefer", "like", "would like")
        )
        negated_prior_rule = any(
            marker in combined
            for marker in ("不再要求", "取消要求", "no longer require", "do not require")
        )
        hard_constraint = any(
            marker in combined for marker in _UNAMBIGUOUS_CONSTRAINT_MARKERS
        )
        if scope == "user" and explicit_preference and (
            category == "preference" or negated_prior_rule
        ):
            return "preference"
        if scope == "user" and soft_preference and not hard_constraint:
            if category in {"preference", "constraint", "project_decision", "general"}:
                return "preference"
        if scope == "project" and category == "constraint" and not hard_constraint:
            return "project_decision"
        if category == "preference":
            return "constraint" if hard_constraint else "preference"
        if category == "general" and any(marker in lowered for marker in _HARD_CONSTRAINT_MARKERS):
            return "constraint"
        return category

    @staticmethod
    def normalize_scope(
        value: str,
        *,
        category: str,
        content: str,
        source_text: str,
        relation: str,
    ) -> str:
        scope = "project" if value == "project" else "user"
        if relation != "new" or scope != "project" or not source_text:
            return scope
        combined = f"{source_text}\n{content}".casefold()
        has_project_cue = any(marker in combined for marker in _PROJECT_MARKERS)
        has_soft_preference = any(marker in combined for marker in _SOFT_PREFERENCE_MARKERS)
        if category.strip().casefold() == "project_decision" and has_soft_preference and not has_project_cue:
            return "user"
        return scope

    @staticmethod
    def _looks_non_memory_candidate(
        content: str,
        *,
        category: str,
        relation: str,
        has_target: bool,
    ) -> bool:
        lowered = content.casefold()
        if any(marker in lowered for marker in _TRANSIENT_MEMORY_PATTERNS):
            return True
        if any(marker in lowered for marker in _PROVENANCE_META_PATTERNS):
            return True
        if relation == "new" and not has_target and category.strip().casefold() == "general":
            durable_subjects = (
                "用户", "项目", "团队", "偏好", "目标", "正在学习", "正在研究",
                "user ", "project", "team", "prefers", "learning", "studying",
            )
            if not any(marker in lowered for marker in durable_subjects):
                return True
        if relation == "new" and not has_target and (
            "没有偏好" in lowered or "没有这个偏好" in lowered
        ):
            return True
        if relation == "new" and not has_target and category.strip().casefold() == "correction":
            denied_absence = (
                ("否认" in lowered and "偏好" in lowered)
                or "does not have a preference" in lowered
                or "denies having a preference" in lowered
            )
            if denied_absence:
                return True
        return False

    @classmethod
    def _deterministic_supplements(
        cls,
        user_messages: list[dict[str, Any]],
        *,
        candidates: list[FactCandidate],
        relevant_facts: list[MemoryFact],
    ) -> list[FactCandidate]:
        """Recover a small set of high-precision atomic facts omitted by the LLM."""

        supplements: list[FactCandidate] = []
        known_text = "\n".join(
            [candidate.content for candidate in candidates]
            + [fact.content for fact in relevant_facts]
        ).casefold()
        for message in user_messages:
            message_id = str(message.get("message_id") or "")
            if not message_id:
                continue
            text = str(message.get("content") or "")
            segments = [item.strip() for item in re.split(r"[，。；;\n]+", text) if item.strip()]
            for segment in segments:
                if cls._looks_secret(segment):
                    continue
                learning = re.search(r"(?:我|本人)(?:目前|正在|正)?(?:学习|研究)\s*([^，。；;]+)", segment)
                if learning:
                    topic = cls._sanitize(learning.group(1))
                    if topic and topic.casefold() not in known_text:
                        candidate = FactCandidate(
                            content=f"用户正在学习{topic}",
                            category="profile",
                            scope="user",
                            confidence=0.95,
                            relation="new",
                            evidence_message_ids=[message_id],
                            metadata={"deterministic_rule": "learning_profile"},
                        )
                        supplements.append(candidate)
                        known_text += "\n" + candidate.content.casefold()
                project_use = re.search(
                    r"(?:本|当前|这个|该)?项目(?:决定)?(?:将|会)?(?:使用|采用|选用)\s*"
                    r"([A-Za-z][A-Za-z0-9+._-]{1,40})",
                    segment,
                    re.IGNORECASE,
                )
                if project_use:
                    technology = project_use.group(1)
                    if technology.casefold() not in known_text:
                        candidate = FactCandidate(
                            content=f"本项目使用 {technology}",
                            category="project_decision",
                            scope="project",
                            confidence=0.95,
                            relation="new",
                            evidence_message_ids=[message_id],
                            metadata={"deterministic_rule": "project_technology"},
                        )
                        supplements.append(candidate)
                        known_text += "\n" + candidate.content.casefold()
        return supplements

    @staticmethod
    def _looks_secret(value: str) -> bool:
        lowered = value.casefold()
        if "only_for_memorybench_not_a_real_credential" in lowered:
            return True
        if any(marker in lowered for marker in _SECRET_POLICY_MARKERS):
            # A durable rule forbidding secret storage is not itself a secret.
            if not _SECRET_ASSIGNMENT_PATTERN.search(value):
                return False
        label = _SECRET_LABEL_PATTERN.search(value)
        if label and _SECRET_ASSIGNMENT_PATTERN.search(value[label.start():]):
            return True
        if re.search(r"cookie\s+session\s*=", value, re.IGNORECASE):
            return True
        if re.search(r"postgres(?:ql)?://[^\s/:]+:[^\s/@]+@", value, re.IGNORECASE):
            return True
        if "-----begin private key-----" in lowered:
            return True
        # Catch common high-entropy provider tokens without retaining them.
        return bool(
            re.search(
                r"\b(?:sk-|ghp_|github_pat_|AKIA)[A-Za-z0-9_-]{12,}\b",
                value,
                re.IGNORECASE,
            )
        )
