from engine.maps.chat_map import ChatMap
from engine.maps.url_classify_map import URLClassifyMap
from engine.maps.judge_map import JudgeMap

COMPLETIONS_MAPS = {
    "chat": ChatMap,
    "url_classify": URLClassifyMap,
    "judge": JudgeMap,
}
