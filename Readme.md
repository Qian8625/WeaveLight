# 添加工具
1. 修改启动配置文件，在 all_service_example_local.yaml 列表的末尾，添加新工具的启动配置
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
    # (2) 配置超时时间（可以不配置）
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
 
3. 采用代码 /home/ubuntu/01_Code/OpenEarthAgent/scripts/tools_test/tools_single_test.py 测试工具效果（注意参考tools的 input和output修改输入输出）
    # 测试成功之后，将对应的模块添加到 scripts/tools_test/tools_test.py 中，补全all工具的测试代码
4. 在 /home/ubuntu/01_Code/OpenEarthAgent/tool_server/tf_eval/utils/rs_agent_prompt.py 中添加模块描述，让Agent理解模块的效果和输入输出
    # 
5. 在 app/app_new.py文件中添加相关的示例进行测试（保存图片为png格式，tiff格式有点问题）
    # 

