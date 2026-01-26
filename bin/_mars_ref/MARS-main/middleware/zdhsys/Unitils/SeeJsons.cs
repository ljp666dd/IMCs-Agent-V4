using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using zdhsys.Bean;
using zdhsys.entity;

namespace zdhsys.Unitils
{
    public class SeeJsons
    {
        public List<List<List<PrintCmd>>> getJson(int flowId, string str)
        {
            List<FlowModel> fms = SqlHelper.GetFlowModelInfo();
            FlowModel fm = fms.Find(x => x.Id == flowId);
            // 1.先查看原流程。把包硅反应这个切分开来。
            List<FlowInfo> infos = JsonConvert.DeserializeObject<List<FlowInfo>>(fm.FlowJson);
            // 判断包硅反应分割几个小流程。一般2个。这里做动态分割
            List<int> bg = new List<int>();
            for (int i = 0; i < infos.Count; i++)
            {
                if (infos[i].tempTxt == "包硅反应")
                {
                    bg.Add(i);
                    Console.WriteLine("包硅反应 index=" + i);
                }
            }
            // 配方数据
            List<string> vs = JsonConvert.DeserializeObject<List<string>>(str);
            // 第一个瓶子的流程。 其它重复。
            List<DevInfo> di = JsonConvert.DeserializeObject<List<DevInfo>>(vs[0]);
            List<List<int>> vvs = new List<List<int>>();
            // 如果没有包硅反应，那么这个流程图就是整个流程。
            if (bg.Count == 0)
            {
                List<int> vs1 = new List<int>();
                for (int i = 0; i < di.Count; i++)
                {
                    vs1.Add(i);
                }
                vs1.Add(di.Count);
                vvs.Add(vs1);
            }
            else
            {
                // 2.首先是分几个小流程
                for (int i = 0; i <= bg.Count; i++)
                {
                    if (i == 0)
                    {
                        List<int> vs1 = new List<int>();
                        for (int k = 0; k < di.Count; k++)
                        {
                            if (k < bg[i])
                            {
                                vs1.Add(k);
                            }
                        }
                        vvs.Add(vs1);
                    }
                    else if (i < bg.Count) //中间段
                    {
                        List<int> vs1 = new List<int>();
                        for (int k = 0; k < di.Count; k++)
                        {
                            if (k > bg[i - 1] && k < bg[i])
                            {
                                vs1.Add(k);
                                //Console.WriteLine("包硅反应 index=" + i);
                            }
                        }
                        vvs.Add(vs1);
                    }
                    else if (i == bg.Count) //最后段
                    {
                        List<int> vs1 = new List<int>();
                        for (int k = 0; k < di.Count; k++)
                        {
                            if (k > bg[i - 1])
                            {
                                vs1.Add(k);
                                //Console.WriteLine("包硅反应 index=" + i);
                            }
                        }
                        vvs.Add(vs1);
                    }
                }
            }
            Console.WriteLine("流程段：" + vvs.Count);
            // 设备列表
            List<DeviceInfoModel> devices = SqlHelper.GetDeviceInfo();
            // 先找出包硅反应的设备组
            GlobalEnum.UnitDeviceType udt = GlobalEnum.UnitDeviceType.包硅反应设备;
            int index = Array.IndexOf(Enum.GetValues(typeof(GlobalEnum.UnitDeviceType)), udt);
            // 包硅设备
            DeviceInfoModel baogui = null;
            // 包硅反应的点位坐标
            DevicePoint dpt = null;
            // 先找出包硅设备。
            baogui = devices.Find(x => x.DeviceGroupId == index.ToString());
            if (baogui != null && !string.IsNullOrEmpty(baogui.PointJson))
            {
                dpt = JsonConvert.DeserializeObject<DevicePoint>(baogui.PointJson);
                Console.WriteLine("包硅反应点位坐标不为空。");
            }

            List<List<List<PrintCmd>>> pppc = new List<List<List<PrintCmd>>>();
            for (int i = 0; i < vvs.Count; i++)
            {
                List<List<PrintCmd>> ppc = new List<List<PrintCmd>>();
                for (int k = 0; k < vs.Count; k++)
                {
                    List<PrintCmd> pc = new List<PrintCmd>();
                    for (int j = 0; j < vvs[i].Count; j++)
                    {
                        List<DevInfo> dif = JsonConvert.DeserializeObject<List<DevInfo>>(vs[k]);
                        for (int n = 0; n < dif.Count; n++)
                        {
                            if (vvs[i][j] == n)
                            {
                                if (n < infos.Count)
                                {
                                    string src = "";
                                    if (infos[n].DeviceId == baogui.Id)
                                    {
                                        //包硅设备. index 从0开始。但是点位从1开始。这里+1
                                        src = getDpoint(dpt, k + 1);
                                    }
                                    else
                                    {
                                        DeviceInfoModel ddim = devices.Find(x => x.Id == infos[n].DeviceId);
                                        src = (int)ddim.X + "";
                                    }

                                    // 0抓=2  1接=6+7 2放=3 3=放+工作+抓
                                    if (infos[n].Cmd == 0)
                                    {
                                        Console.WriteLine($"设备ID={dif[n].Id} 流程Cmd= 2 {src} 0");
                                        PrintCmd pcd = new PrintCmd
                                        {
                                            cmd = "2 " + src + " 0",
                                            opt = 0,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId
                                        };
                                        pc.Add(pcd);
                                        PrintCmd pcd2 = new PrintCmd
                                        {
                                            opt = 1,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId,
                                            devInfo = dif[n]
                                        };
                                        if (!isZero(pcd2.devInfo))
                                        {
                                            pc.Add(pcd2);
                                        }
                                    }
                                    else if (infos[n].Cmd == 1)
                                    {
                                        Console.WriteLine($"设备ID={dif[n].Id} 流程Cmd= 6 {src} 0   sleep(30)流程Cmd= 7 {src} 0");
                                        PrintCmd pcd = new PrintCmd
                                        {
                                            cmd = "6 " + src + " 0",
                                            opt = 0,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId
                                        };
                                        PrintCmd pcd2 = new PrintCmd
                                        {
                                            opt = 1,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId,
                                            devInfo = dif[n]
                                        };
                                        PrintCmd pcd3 = new PrintCmd
                                        {
                                            cmd = "7 " + src + " 0",
                                            opt = 0,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId
                                        };
                                        if (!isZero(pcd2.devInfo))
                                        {
                                            pc.Add(pcd);
                                            pc.Add(pcd2);
                                            pc.Add(pcd3);
                                        }
                                    }
                                    else if (infos[n].Cmd == 2)
                                    {
                                        Console.WriteLine($"设备ID={dif[n].Id} 流程Cmd= 3 {src} 0");
                                        PrintCmd pcd = new PrintCmd
                                        {
                                            cmd = "3 0 " + src,
                                            opt = 0,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId
                                        };
                                        pc.Add(pcd);
                                    }
                                    else if (infos[n].Cmd == 3)
                                    {
                                        PrintCmd pcd = new PrintCmd
                                        {
                                            cmd = "3 0 " + src,
                                            opt = 0,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId
                                        };
                                        pc.Add(pcd);
                                        PrintCmd pcd2 = new PrintCmd
                                        {
                                            opt = 1,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId,
                                            devInfo = dif[n]
                                        };
                                        if (!isZero(pcd2.devInfo))
                                        {
                                            pc.Add(pcd2);
                                        }
                                        PrintCmd pcd3 = new PrintCmd
                                        {
                                            cmd = "2 " + src + " 0",
                                            opt = 0,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId
                                        };
                                        pc.Add(pcd3);
                                        Console.WriteLine($"设备ID={dif[n].Id} 流程Cmd= 3 {src} 0");
                                        Console.WriteLine($"开始工作=============================");
                                        Console.WriteLine($"设备ID={dif[n].Id} 流程Cmd= 2 {src} 0");
                                    }
                                }
                                else
                                {
                                    // 特别注意。这里使用的是  infos  n-1 的，因为这个就是最后一个流程，取 cmd2。 
                                    string src = "";
                                    if (infos[n - 1].DeviceId2 == baogui.Id)
                                    {
                                        //包硅设备
                                        src = getDpoint(dpt, k + 1);
                                    }
                                    else
                                    {
                                        DeviceInfoModel ddim = devices.Find(x => x.Id == infos[n - 1].DeviceId2);
                                        src = (int)ddim.X + "";
                                    }

                                    // 0抓=2  1接=6+7 2放=3
                                    if (infos[n - 1].Cmd2 == 0)
                                    {
                                        Console.WriteLine($"设备ID={dif[n].Id} 流程Cmd= 2 {src} 0");
                                        PrintCmd pcd = new PrintCmd
                                        {
                                            cmd = "2 " + src + " 0",
                                            opt = 0,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId
                                        };
                                        pc.Add(pcd);
                                        PrintCmd pcd2 = new PrintCmd
                                        {
                                            opt = 1,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId,
                                            devInfo = dif[n]
                                        };
                                        if (!isZero(pcd2.devInfo))
                                        {
                                            pc.Add(pcd2);
                                        }
                                    }
                                    else if (infos[n - 1].Cmd2 == 1)
                                    {
                                        Console.WriteLine($"设备ID={dif[n].Id} 流程Cmd= 6 {src} 0   sleep(30)流程Cmd= 7 {src} 0");
                                        PrintCmd pcd = new PrintCmd
                                        {
                                            cmd = "6 " + src + " 0",
                                            opt = 0,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId
                                        };
                                        PrintCmd pcd2 = new PrintCmd
                                        {
                                            opt = 1,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId,
                                            devInfo = dif[n]
                                        };
                                        PrintCmd pcd3 = new PrintCmd
                                        {
                                            cmd = "7 " + src + " 0",
                                            opt = 0,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId
                                        };
                                        // 如果这个加粉液的设备参数都是0，就直接跳过这个设备
                                        if (!isZero(pcd2.devInfo))
                                        {
                                            pc.Add(pcd);
                                            pc.Add(pcd2);
                                            pc.Add(pcd3);
                                        }
                                    }
                                    else if (infos[n - 1].Cmd2 == 2)
                                    {
                                        Console.WriteLine($"设备ID={dif[n].Id} 流程Cmd= 3 {src} 0");
                                        PrintCmd pcd = new PrintCmd
                                        {
                                            cmd = "3 0 " + src,
                                            opt = 0,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId
                                        };
                                        pc.Add(pcd);
                                    }
                                    else if (infos[n - 1].Cmd2 == 3)
                                    {
                                        Console.WriteLine($"设备ID={dif[n].Id} 流程Cmd= 3 0 {src}");
                                        Console.WriteLine($"开始称重");
                                        Console.WriteLine($"设备ID={dif[n].Id} 流程Cmd= 2 {src} 0");
                                        PrintCmd pcd = new PrintCmd
                                        {
                                            cmd = "3 0 " + src,
                                            opt = 0,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId
                                        };
                                        pc.Add(pcd);
                                        PrintCmd pcd2 = new PrintCmd
                                        {
                                            opt = 1,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId,
                                            devInfo = dif[n]
                                        };
                                        if (!isZero(pcd2.devInfo))
                                        {
                                            pc.Add(pcd2);
                                        }
                                        PrintCmd pcd3 = new PrintCmd
                                        {
                                            cmd = "2 " + src + " 0",
                                            opt = 0,
                                            device = devices.Find(x => x.Id == dif[n].Id).DeviceId
                                        };
                                        pc.Add(pcd3);
                                    }
                                }
                                break;
                            }
                        }
                    }
                    ppc.Add(pc);
                    Console.WriteLine("Json=" + JsonConvert.SerializeObject(pc));
                    Console.WriteLine("开始阶段内瓶子循环：++++++++++++++++++++++++");
                }
                Console.WriteLine("JsonList=" + JsonConvert.SerializeObject(ppc));
                Console.WriteLine(" 开始下一段循环:-------------------------------------");
                pppc.Add(ppc);
            }
            Console.WriteLine("JsonListList=" + JsonConvert.SerializeObject(pppc));
            return pppc;
        }

        private bool isZero(DevInfo dio)
        {
            for (int i = 0; i < dio.Dfms.Count; i++)
            {
                if (int.TryParse(dio.Dfms[i].FieldsContent, out int ret))
                {
                    if (ret != 0)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        private string getDpoint(DevicePoint dpt, int index)
        {
            string src = "";
            switch (index)
            {
                case 1:
                    src = dpt.p1;
                    break;
                case 2:
                    src = dpt.p2;
                    break;
                case 3:
                    src = dpt.p3;
                    break;
                case 4:
                    src = dpt.p4;
                    break;
                case 5:
                    src = dpt.p5;
                    break;
                case 6:
                    src = dpt.p6;
                    break;
                case 7:
                    src = dpt.p7;
                    break;
                case 8:
                    src = dpt.p8;
                    break;
                case 9:
                    src = dpt.p9;
                    break;
                case 10:
                    src = dpt.p10;
                    break;
                case 11:
                    src = dpt.p11;
                    break;
                case 12:
                    src = dpt.p12;
                    break;
                default:
                    break;
            }
            return src;
        }
    }
}
