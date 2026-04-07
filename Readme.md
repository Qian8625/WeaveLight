# 添加工具
1. 修改启动配置文件，在 tool_worker_config 列表的末尾，添加新工具的启动配置
    例如：
    - SARDualFreqDiff:
        worker_name: SARDualFreqDiff
        job_name: sar_dual_freq_diff
        calculate_type: control
        cmd:
            script-addr: ./online_workers/SARDualFreqDiff_worker.py  # 你的脚本路径
            port: 20024  # 独立的端口
            host: "0.0.0.0"


2. 修改 Tool Manager 校验逻辑
    代码路径：tool_server/tool_workers/tool_manager/base_manager.py
    # (1) 添加到检查列表
    miss_tool = []
        for tool in ['RegionAttributeDescription', 'OCR', 'DrawBox', 'Plot', 'AddPoisLayer', 'GetAreaBoundary', 'ChangeDetection', 'Solver', 'SegmentObjectPixels', 'AddText', 'ObjectDetection', 'GoogleSearch', 'BaseModel', 'CountGivenObject', 'Calculator', 'SARDualFreqDiff']: # <- 添加在这里
            if tool not in self.available_online_tools:
                miss_tool.append(tool)
    # (2) 配置超时时间
    def call_tool(self,tool_name,params):
        if tool_name in ["AddPoisLayer","ComputeDistance"]:
            timeout_sec = 180
        elif tool_name in ["AddIndexLayer"]:
            timeout_sec = 300
        # 将你的工具加入到拥有 120秒（或更长）超时限制的列表中
        elif tool_name in ["ChangeDetection", "GetAreaBoundary", "SARDualFreqDiff"]: # <- 添加在这里
            timeout_sec = 120
        else:
            timeout_sec = 60  # timeout per attempt