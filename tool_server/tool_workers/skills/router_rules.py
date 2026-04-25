ROUTER_RULES = {
    "TargetLocateMeasureSkill": {
        "keywords": [
            "locate", "find", "detect", "box", "bbox", "measure",
            "distance", "area", "segment",
            "定位", "找出", "检测", "框选", "面积", "距离", "分割"
        ],
        "negative_keywords": [
            "sar", "rgb and sar", "cross-modal", "change", "geotiff", "poi",
            "变化", "跨模态", "双时相", "地理", "兴趣点"
        ],
        "preferred_inputs": ["primary_image"],
        "required_modalities": [],
    },
    "SARTargetLocateMeasureSkill": {
        "keywords": [
            "sar", "sar image", "radar", "ship", "aircraft",
            "sar 图像", "雷达图像", "船只", "飞机", "面积", "距离", "框选"
        ],
        "negative_keywords": [
            "rgb and sar", "cross-modal", "change", "geotiff", "poi",
            "跨模态", "变化", "双时相", "地理", "兴趣点"
        ],
        "preferred_inputs": ["primary_image"],
        "required_modalities": ["sar"],
    },
    "TargetAttributeSkill": {
        "keywords": [
            "color", "attribute", "same", "different", "compare",
            "describe and count", "red", "blue", "orientation",
            "颜色", "属性", "比较", "是否相同", "描述", "统计"
        ],
        "negative_keywords": [
            "sar", "change", "geotiff", "poi", "distance only", "area only",
            "变化", "地理", "兴趣点"
        ],
        "preferred_inputs": ["primary_image"],
        "required_modalities": [],
    },
    "ConditionalCountSkill": {
        "keywords": [
            "count", "how many", "docked", "parked", "facing east", "facing west",
            "统计", "数量", "多少", "靠泊", "停放", "朝东", "朝西"
        ],
        "negative_keywords": [
            "change", "geotiff", "poi", "rgb and sar", "cross-modal",
            "变化", "兴趣点", "跨模态"
        ],
        "preferred_inputs": ["primary_image"],
        "required_modalities": [],
    },
    "MultConfirmSkill": {
        "keywords": [
            "rgb and sar", "sar confirm", "confirm with sar", "cross-modal", "fusion",
            "rgb", "sar", "compare modalities",
            "用sar确认", "跨模态", "融合", "rgb中", "sar中"
        ],
        "negative_keywords": [
            "geotiff", "poi", "change",
            "兴趣点", "地理", "变化"
        ],
        "preferred_inputs": ["primary_image", "time1_image"],
        "required_modalities": ["rgb", "sar"],
    },
    "ChangeSummarySkill": {
        "keywords": [
            "change", "compare two images", "before and after", "new ships",
            "disappear", "moved", "transferred", "damage", "expansion",
            "变化", "两期", "前后", "新增", "消失", "转移", "损毁", "扩建"
        ],
        "negative_keywords": [
            "geotiff", "poi", "single image only",
            "兴趣点", "单张图"
        ],
        "preferred_inputs": ["time1_image", "time2_image"],
        "required_modalities": ["time_series"],
    },
    "GeoTIFFPoiExploreSkill": {
        "keywords": [
            "geotiff", "poi", "hospital", "museum", "mall", "around", "surrounding",
            "exist", "how many poi", "visualize poi",
            "geotiff", "兴趣点", "医院", "博物馆", "商场", "周边", "是否存在", "可视化"
        ],
        "negative_keywords": [
            "distance between poi",
            "nearest poi",
            "closest pair",
            "compute distance",
            "最近距离",
            "poi距离",
            "距离"
        ],
        "preferred_inputs": ["primary_image"],
        "required_modalities": ["geotiff"],
    },
    "GeoTIFFPoiDistanceSkill": {
        "keywords": [
            "geotiff", "poi distance", "nearest", "closest",
            "distance between poi", "closest pair", "compute distance",
            "distance between", "between the closest pair",
            "最近距离", "poi距离", "两类poi", "最近", "距离"
        ],
        "negative_keywords": [
            "change", "sar", "single target detection",
            "变化", "雷达", "目标检测"
        ],
        "preferred_inputs": ["primary_image"],
        "required_modalities": ["geotiff"],
    },
}