using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO.Ports;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Forms.VisualStyles;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Threading;
using zdhsys.Bean;
using zdhsys.entity;
using zdhsys.Popup;
using zdhsys.Unitils;



namespace zdhsys.UI
{
    /// <summary>
    /// UCHome.xaml 的交互逻辑
    /// </summary>
    public partial class UCHome : UserControl
    {
        public UCHome()
        {
            InitializeComponent();
            InitUI();
        }
        private static UdpClient udpClient;
        private static List<RobotInfoModel> vs;
        private static string robotIP = "";
        private static int robotPort = 2368;
        private static Thread thUdp = null;
        private static Options opts = null;
        private static Thread thWork = null;
        private static bool IsStop = false;
        private static List<DeviceInfoModel> devices = null;
        Dev_CZ czdevice;
        Dev_BG bgdevice;
        //加液
        private static SerialPort spFY;
        private void InitUI()
        {
            Btn_FlowLoad.Click += Btn_FlowLoad_Click;
            Btn_run.Click += Btn_run_Click;
            Btn_Stop.Click += Btn_Stop_Click;

            czdevice = new Dev_CZ();//称重设备

            //   bgdevice = new Dev_BG();//包硅设备


            timer1 = new System.Windows.Forms.Timer();
            timer1.Tick += timerupdate;
            timer1.Interval = 1000;
            timer1.Start();

        }
        //add by jgl
        System.Windows.Forms.Timer timer1;
        int tk = 0;
        private void timerupdate(object sender, EventArgs e)
        {
            int gdpose;
            if (ht != null)
            {
                gdpose = ht.guideway;
            }
            else
            {
                gdpose = 0;
            }
            //tk++;
            double wid = this.bd_robot.ActualWidth;
            double y = wid * (3000000 - gdpose) / 6000000;
            Thickness robotmar = dev_robot.Margin;
            dev_robot.Margin = new Thickness(y, 0, 0, 0);
            // if (tk > 50) tk = 0;
            //  Point loc = this.dev_robot

            //  double Ld = this.bd_robot.Margin.Left;
            //   double Rd = this.bd_robot.Margin.Right;

            //   Console.WriteLine(y.ToString() + " "+ gdpose.ToString() + " " + wid.ToString());
        }

        /// <summary>
        /// 停止
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Btn_Stop_Click(object sender, RoutedEventArgs e)
        {
            IsStop = true;
        }

        /// <summary>
        /// 开始运行
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private async void Btn_run_Click(object sender, RoutedEventArgs e)
        {
            if (opts == null)
            {
                Point p = GetParentWindows.GetPoint(this);
                MessageInfo msg = new MessageInfo(p)
                {
                    Width = ActualWidth,
                    Height = ActualHeight
                };
                msg.SetUI("请先加载配方!");
                _ = msg.ShowDialog();
                return;
            }
            /*   if (thWork == null)
               {
                   IsStop = false;
                   /* thWork = new Thread(new ThreadStart(Run))
                    //  thWork = new Thread(new ThreadStart(RunDemo))//for sty caitaikuang
                 //  
                    {
                        IsBackground = true
                    };
                    thWork.Start();*/
            /*     await Task.Run(() => RunDemo());

                 //禁用运行按钮。和载入按钮
                 Btn_FlowLoad.IsEnabled = false;
                 Btn_run.IsEnabled = false;
                 Btn_run.SetUI("#F0F0F0", "#F0F0F0", "\\Image\\安全运行.png", "#FFFFFF", "运行");
                 Console.WriteLine("运行");
             }*/
            if (!IsStop) // 确保任务没有在运行中再次启动
            {
                IsStop = true;

                try
                {
                    //禁用运行按钮和载入按钮
                    Btn_FlowLoad.IsEnabled = false;
                    Btn_run.IsEnabled = false;
                    Btn_run.SetUI("#F0F0F0", "#F0F0F0", "\\Image\\安全运行.png", "#FFFFFF", "运行");
                    Console.WriteLine("运行");

                    await Task.Run(() => RunDemo());
                }
                finally
                {
                    //无论是否发生异常，都会执行此代码块，用于确保UI恢复
                    IsStop = false;
                    Btn_FlowLoad.IsEnabled = true;
                    Btn_run.IsEnabled = true;
                    // 可能还需要重置按钮的UI到原始状态
                }
            }
        }
        /// <summary>
        /// 开始执行配方
        /// </summary>
        private void Run()
        {
            SeeJsons sj = new SeeJsons();
            List<List<List<PrintCmd>>> ppc = sj.getJson(opts.FlowId, opts.OptionJson);
            //多个阶段
            //MoveRobot();
            for (int i = 0; i < ppc.Count; i++)
            {
                //单个阶段的12个瓶子。
                List<List<PrintCmd>> pc = ppc[i];
                for (int k = 0; k < pc.Count; k++)
                {
                    // 测试。这里只执行第一个瓶子。 正常工作这个判断要去掉
                    if (k > 0)//0,1,2,3  执行1-4号
                    {
                        break;
                    }
                    //单个瓶子的所有步骤
                    List<PrintCmd> pcd = pc[k];
                    for (int j = 0; j < pcd.Count; j++)
                    {
                        if (pcd[j].opt == 0)
                        {
                            // 判断是否停止. 这里还不是急停
                            if (IsStop)
                            {
                                FinishToUI();
                            }
                            //发送指令给机械臂
                            if (CmdSendUdp(pcd[j]))
                            {
                                Console.WriteLine("机械臂执行成功");
                            }
                            else
                            {
                                Console.WriteLine("机械臂执行失败!!!");
                                bool ret = false;
                                Dispatcher.Invoke(() =>
                                {
                                    Point p = GetParentWindows.GetPoint(this);
                                    MessageInfo3 msg = new MessageInfo3(p)
                                    {
                                        Width = ActualWidth,
                                        Height = ActualHeight
                                    };
                                    msg.SetUI("机械臂执行指令失败,请管理员处理后，选择是否继续!");
                                    _ = msg.ShowDialog();
                                    // 结束当前配方
                                    if (!msg.flag)
                                    {
                                        FinishToUI();
                                        ret = true;
                                        return;
                                    }
                                });
                                if (ret)
                                {
                                    Console.WriteLine("结束当前配方执行。");
                                    return;
                                }
                            }
                        }
                        else if (pcd[j].opt == 1)
                        {
                            Console.WriteLine("控制反应设备，这里暂时不执行。先跳过");
                            DevInfo dev = pcd[j].devInfo;

                            DeviceInfoModel dfm = devices.Find(x => x.Id == dev.Id);
                            Console.WriteLine("设备ID:" + dfm.DeviceId);
                            //获取设备，然后准备拼装数据
                            int typeId = 0;
                            _ = int.TryParse(dfm.DeviceGroupId, out typeId);
                            // 1.处理加粉液设备
                            if (typeId == Array.IndexOf(Enum.GetValues(typeof(GlobalEnum.UnitDeviceType)), GlobalEnum.UnitDeviceType.加粉液设备))
                            {
                                byte did = byte.Parse(dfm.DeviceId);
                                short num = 0;
                                int time = 0;
                                for (int d = 0; d < dev.Dfms.Count; d++)
                                {
                                    if (dev.Dfms[d].FieldsName == "数量")
                                    {
                                        _ = short.TryParse(dev.Dfms[d].FieldsContent, out num);
                                    }
                                    else if (dev.Dfms[d].FieldsName == "时间")
                                    {
                                        _ = int.TryParse(dev.Dfms[d].FieldsContent, out time);
                                    }
                                }
                                // 数量大于0，就需要添加。
                                if (num > 0)
                                {
                                    AddWater1(did, num);
                                    //机械臂等待时间
                                    if (time > 0)
                                    {
                                        Thread.Sleep(time * 1000);
                                    }
                                }
                            }
                            else if (typeId == Array.IndexOf(Enum.GetValues(typeof(GlobalEnum.UnitDeviceType)), GlobalEnum.UnitDeviceType.包硅反应设备))
                            {
                                //其它设备。。
                                int num = 0;
                                for (int d = 0; d < dev.Dfms.Count; d++)
                                {
                                    if (dev.Dfms[d].FieldsName == "指令")
                                    {
                                        _ = int.TryParse(dev.Dfms[d].FieldsContent, out num);
                                        break;
                                    }
                                }
                                Console.WriteLine($"the mem op of 包硅设备 is {num}");
                                if (num == 1)
                                {

                                }

                            }
                            else if (typeId == Array.IndexOf(Enum.GetValues(typeof(GlobalEnum.UnitDeviceType)), GlobalEnum.UnitDeviceType.称量设备))
                            {
                                //其它设备。。
                                int num = 0;
                                for (int d = 0; d < dev.Dfms.Count; d++)
                                {
                                    if (dev.Dfms[d].FieldsName == "指令")
                                    {
                                        _ = int.TryParse(dev.Dfms[d].FieldsContent, out num);
                                        break;
                                    }
                                }
                                Console.WriteLine($"the mem op of 称重设备 is {num}");
                                // Console.WriteLine(num);
                                // 
                                if (num == 1)//测量磁珠
                                {
                                    czdevice.sendcmd_initial_until_start();
                                    Thread.Sleep(1 * 1000);
                                    czdevice.sendcmd_Mem2_until_start(k + 1);
                                    while (true)
                                    {
                                        Thread.Sleep(3 * 1000);
                                        if (czdevice.latest_dev_state == 8)
                                        {
                                            int pod = czdevice.latest_pod;
                                            double val = czdevice.latest_val;
                                            Console.WriteLine("data received!");
                                            break;  
                                        }
                                    }
                                    Console.WriteLine("send mem2 cmd!");
                                    Thread.Sleep(5000);
                                }
                                else if (num == 2)//添加磁核
                                {
                                    Console.WriteLine("send ADD cmd!");
                                    /*    czdevice.sendcmd_initial_until_start();
                                        Thread.Sleep(1 * 1000);
                                        czdevice.sendcmd_Add_until_start(k+1, 2);
                                        while (true)
                                         {
                                             Thread.Sleep(3 * 1000);
                                             if (czdevice.latest_dev_state == 8)
                                             {
                                                 //int pod = czdevice.latest_pod;
                                                 //double val = czdevice.latest_val;
                                                 Console.WriteLine("Add finished!");
                                                 break;
                                             }
                                         }
                                    */
                                    Thread.Sleep(2000);
                                }
                                else if (num == 3)//测量磁核
                                {

                                }
                                else { Console.WriteLine("no op at mem"); }
                            }
                        }
                    }
                }
            }

            PrintCmd homecmd = new PrintCmd();
            homecmd.cmd = "5 0 0";
            CmdSendUdp(homecmd);


            FinishToUI();
        }
        /// <summary>
        /// 恢复UI的可操作性
        /// 

        private async Task startMem(int id=0)
        {
            czdevice.sendcmd_initial_until_start();
            await Task.Delay(1000);
           // Thread.Sleep(1 * 1000);
            czdevice.sendcmd_Mem2_until_start(1);
            await Task.Delay(1000);
        }
        private async Task waitForMem()
        {
            while (true)
            {
                //Thread.Sleep(3 * 1000);
                await Task.Delay(1000);
                if (czdevice.latest_dev_state == 8)
                {
                    int pod = czdevice.latest_pod;
                    double val = czdevice.latest_val;
                    Console.WriteLine("data received!");
                    break;
                }
            }
        }

        private async Task<bool> getPod(int id, CancellationToken cts)
        {
            PrintCmd Rcmd = new PrintCmd();
            // 设置并发送初始化命令
            Rcmd.cmd = "2 "+id.ToString()+" 0";
            Console.WriteLine("Sending robot command...");
            if (!await CmdSendUdp_1Async(Rcmd, cts))
            {
                Console.WriteLine("robot command failed.");
                FinishToUI();
                return false;
            }
            return true;
        }

        private async Task<bool> putPod(int id, CancellationToken cts)
        {
            PrintCmd Rcmd = new PrintCmd();
            // 设置并发送初始化命令
            Rcmd.cmd = "3 " +"0 "+ id.ToString();
            Console.WriteLine("Sending robot command...");
            if (!await CmdSendUdp_1Async(Rcmd, cts))
            {
                Console.WriteLine("robot command failed.");
                FinishToUI();
                return false;
            }
            return true;
        }

        private void addwaterbyID(int id, int vol)
        {
            RobotID2StationID.getStationID(id, out int staID);
            byte did = (byte)staID;
            short vv = (short)vol;
            //yte.Parse(staID);
            AddWater1(did, vv);
        }

        private async Task<bool> jiePodatID(int id, int vol, CancellationToken cts)
        {
            PrintCmd Rcmd = new PrintCmd();
            // 设置并发送初始化命令
            Rcmd.cmd = "6 " + id.ToString() + " 0";
            Console.WriteLine("Sending robot jie command...");
            Console.WriteLine(Rcmd.cmd);
               if (!await CmdSendUdp_1Async(Rcmd, cts))
               {
                   Console.WriteLine("robot command failed.");
                   FinishToUI();
                   return false;
               }
               

            RobotID2StationID.getStationID(id, out int staID);
            byte did = (byte)staID;
            short vv = (short)vol;
                //yte.Parse(staID);
            AddWater1(did, vv);

            await Task.Delay(vv*1000+3000);

            // 设置并发送初始化命令
            Rcmd.cmd = "7 " + id.ToString() + " 0";
            Console.WriteLine("Sending robot hui command...");
         //   Console.WriteLine("Sending robot jie command...");
           Console.WriteLine(Rcmd.cmd);
            if (!await CmdSendUdp_1Async(Rcmd, cts))
            {
                Console.WriteLine("robot command failed.");
                FinishToUI();
                return false;
            }
            

            return true;

        }


        // sty's experiment
        private async void RunDemo()
        {
            PrintCmd homecmd = new PrintCmd();
            var cts = new CancellationTokenSource();
            var app = (App)Application.Current;
            var experimentData = app.experimentData;
         //   Console.WriteLine($"Experiment Name: {experimentData.Materials}");
            // 从experimentData中取数据
            try
            {
                int cyc = 1;
                while(cyc>0)
                {
                    cyc--;/*
                    await getPod(26, cts.Token);
                    await jiePodatID(6, 10, cts.Token);
                    await jiePodatID(7, 10, cts.Token);
                    await putPod(10, cts.Token);
                    await startMem();
                    await Task.Delay(5000);*/

                       await getPod(23, cts.Token);
                       //await putPod(25, cts.Token);
                       await jiePodatID(8, 1, cts.Token);
                       await jiePodatID(9, 1, cts.Token);
                       await putPod(25, cts.Token);
                       // 异步启动工作站并等待完成
                       Console.WriteLine("Starting workstation...");
                       bool ret = await startWorkStation(0);  // 确保这个方法也是异步的
                       if (!ret)
                       {
                           Console.WriteLine(" workstation.errot..");
                           return;
                       }
                       await getPod(25, cts.Token);
                       await putPod(24, cts.Token);

                       await getPod(22, cts.Token);
                       await putPod(25, cts.Token);

                       ret = await startWorkStation(1);
                       if (!ret)
                       {
                           Console.WriteLine(" workstation.errot..");
                           return;
                       }

                       await getPod(25, cts.Token);
                       await putPod(22, cts.Token);

                       await getPod(24, cts.Token);
                       await putPod(23, cts.Token);

                       /*await Task.Delay(5000);
                       await waitForMem();
                       await getPod(10, cts.Token);
                       await putPod(26, cts.Token);*/
                      
                }




                /*       // 设置并发送初始化命令
                       homecmd.cmd = "2 23 0";
                       Console.WriteLine("Sending initialization command...");
                       if (!await CmdSendUdp_1Async(homecmd, cts.Token))
                       {
                           Console.WriteLine("Initialization command failed.");
                           FinishToUI();
                           return;
                       }

                       // 异步启动工作站并等待完成
                       Console.WriteLine("Starting workstation...");
                       await startWorkStation();  // 确保这个方法也是异步的

                       // 发送后续命令
                       homecmd.cmd = "4 0 0";
                       Console.WriteLine("Sending follow-up command...");
                       if (!await CmdSendUdp_1Async(homecmd, cts.Token))
                       {
                           Console.WriteLine("Follow-up command failed.");
                           FinishToUI();
                           return;
                       }

                       homecmd.cmd = "11 0 0";
                       Console.WriteLine("Sending another follow-up command...");
                       if (!await CmdSendUdp_1Async(homecmd, cts.Token))
                       {
                           Console.WriteLine("Another follow-up command failed.");
                           FinishToUI();
                           return;
                       }*/

                // 完成并向UI反馈
                Console.WriteLine("Finishing and updating UI...");
                FinishToUI();
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("The operation was canceled.");
                FinishToUI();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An error occurred in RunDemo: {ex.Message}");
                FinishToUI();
            }
            finally
            {
                // 确保取消令牌源被正确释放
                cts.Dispose();
            }
        }

        /// </summary>
        private void FinishToUI()
        {
            Dispatcher.Invoke(() =>
            {
                // 恢复运行按钮为可用
                Btn_run.IsEnabled = true;
                Btn_FlowLoad.IsEnabled = true;
                Btn_run.SetUI("#0080FF", "#0080FF", "\\Image\\安全运行.png", "#FFFFFF", "运行");
            });
            if (opts != null)
            {
                opts = null;
            }
            if (thWork != null)
            {
                thWork.Abort();
                thWork = null;
            }
            IsStop = true;
        }

        private async void WaitRobotWork()
        {
           // Thread.Sleep(2000);
           await Task.Delay(2000);
            while (true)
            {
                if (ht != null)
                {
                    if (ht.rob_sta == 0)
                    {
                        break;
                    }
                }
                // Thread.Sleep(1000);
                await Task.Delay(1000);
            }
            //  Thread.Sleep(3000);
            await Task.Delay(2000);
        }

        private void AddWater(byte id)
        {
            if (id == 0x01)
            {
                byte[] jycmd = { 0xaa, 0x01, 0x02, 0x06, 0x55 };
                spFY.Write(jycmd, 0, jycmd.Length);
            }
            else if (id == 0x02)
            {
                byte[] jycmd = { 0xaa, 0x02, 0x02, 0x02, 0x55 };
                spFY.Write(jycmd, 0, jycmd.Length);
            }
            else if (id == 0x03)
            {
                byte[] jycmd = { 0xaa, 0x03, 0x02, 0x0f, 0x55 };
                spFY.Write(jycmd, 0, jycmd.Length);
            }
            else if (id == 0x04)
            {
                byte[] jycmd = { 0xaa, 0x04, 0x02, 0x0f, 0x55 };
                spFY.Write(jycmd, 0, jycmd.Length);
            }
        }
        /// <summary>
        /// 加粉液设备ID和数量
        /// </summary>
        /// <param name="id"></param>
        /// <param name="val"></param>
        private void AddWater1(byte id, short val)
        {
            byte[] vs = BitConverter.GetBytes(val);
            byte[] jycmd = { 0xaa, 0x01, id, 0x02, 0x00, vs[1], vs[0], 0x55 };
            spFY.Write(jycmd, 0, jycmd.Length);
        }

        private void InitData()
        {
            //机器人
            if (udpClient == null)
            {
                udpClient = new UdpClient(2368);
                Console.WriteLine("udp init");
                vs = SqlHelper.GetRobotInfo();
                if (vs != null && vs.Count > 0)
                {
                    robotIP = vs[0].RobotIP;
                    robotPort = int.Parse(vs[0].RobotPort);
                    Console.WriteLine("TCP " + robotIP + " : " + robotPort);
                }
                else
                {
                    return;
                }
                //在线程里面进行监听。机械臂的回传信息.
                thUdp = new Thread(new ThreadStart(StartReceiving))
                {
                    IsBackground = true
                };
                thUdp.Start();
            }
            //加粉设备
            if (spFY == null)
            {
                spFY = new SerialPort("COM10", 9600, Parity.None, 8, StopBits.One);
                try
                {
                    spFY.Open();
                }
                catch
                {
                    spFY = null;
                }
            }
        }

        private void Btn_FlowLoad_Click(object sender, RoutedEventArgs e)
        {
            Point p = GetParentWindows.GetPoint(this);
            SelectFlowData sfd = new SelectFlowData(p)
            {
                Width = ActualWidth,
                Height = ActualHeight
            };
            _ = sfd.ShowDialog();
            if (sfd.flag_Save)
            {
                opts = sfd.ofm;
            }
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e)
        {
            devices = SqlHelper.GetDeviceInfo();
            InitData();
            DataToUI();
        }

        private static Heart ht = null;

        public async Task<bool> startWorkStation(int rcpid)
        {
            using (var communicator = new UdpCommunicator("192.168.1.253", 53, "192.168.1.196", 9696,96,rcpid))
            {
                WorkstationManager manager = new WorkstationManager(communicator);
                bool result = await manager.ExecuteWorkstationTaskAsync();
                Console.WriteLine($"Communication result: {(result ? "Success" : "Failed")}");
                return result;
            }
        }
        /// <summary>
        /// 接收UDP信息：心跳
        /// </summary>
        private async void StartReceiving()
        {
            while (true)
            {
                //Console.WriteLine("StartReceiving....");
                UdpReceiveResult result = await udpClient.ReceiveAsync();   // 异步接收数据
                byte[] receivedData = result.Buffer;
                string receivedString = Encoding.UTF8.GetString(receivedData);
                if (string.IsNullOrEmpty(receivedString))
                {
                    continue;
                }
                RobotHeart rh = null;
                ht = null;
                try
                {
                    rh = JsonConvert.DeserializeObject<RobotHeart>(receivedString);
                    if (rh.cmd == 1)
                    {
                        ht = JsonConvert.DeserializeObject<Heart>(rh.data.ToString());
                        //Console.WriteLine("机器人状态:" + ht.rob_sta + " 夹子状态:" + ht.pod_sta);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine("UDP 上送机器人心跳数据解析异常:" + ex.ToString());
                }
                if (rh != null && ht != null)
                {
                    // 在UI线程中更新UI
                    Dispatcher.Invoke(() =>
                    {
                        //更新UI
                        //Console.WriteLine(receivedString);
                        if (ht.rob_sta == 0)
                        {
                            dev_robot.SetUI("导轨机器人", "空闲", "\\Image\\机器人.png");
                        }
                        else
                        {
                            dev_robot.SetUI("导轨机器人", "执行中", "\\Image\\机器人.png");
                        }
                        dev_robot.ht = ht;
                        dev_robot.rh = rh;
                    });
                }
            }
        }
        /// <summary>
        /// 指令发送，且等待执行，和执行完成后返回
        /// 
        private bool CmdSendUdp_1(PrintCmd cmd)
        {
            Console.WriteLine("current cmd is: " + cmd.cmd);

            // 拼装发送内容
            RobotHeart rh = new RobotHeart
            {
                point = 1,
                cmd = int.Parse(cmd.cmd.Split(' ')[0]),
                data = new MoveData
                {
                    ID_src = int.Parse(cmd.cmd.Split(' ')[1]),
                    ID_dst = int.Parse(cmd.cmd.Split(' ')[2])
                }
            };

            // 发送UDP消息
            SendUdp(rh);
            Console.WriteLine("发送CMD=" + cmd.cmd);

            // 等待1秒后，机械臂进入工作状态，这里做超时处理
            Task.Delay(1000).Wait(); // 使用 Task.Delay 并等待其完成

            int cnt = 0;
            while (true)
            {
                cnt++;
                Task.Delay(500).Wait(); // 使用 Task.Delay 并等待其完成
                if (ht != null && ht.rob_sta == 1)
                {
                    Console.WriteLine("机械臂执行指令中...");
                    break;
                }

                if (cnt > 10)
                {
                    Console.WriteLine("robot start time out!");
                    return false;
                }
            }

            // 等待机械臂返回执行完成，再等3秒后返回主线程
            WaitRobotWork();
            return true;
        }

        private async Task WaitRobotWorkAsync(CancellationToken cancellationToken)
        {
            // 等待2秒让机器人开始工作
            await Task.Delay(2000, cancellationToken);

            while (!cancellationToken.IsCancellationRequested)
            {
                if (ht != null && ht.rob_sta == 0)
                {
                    break;
                }

                // 每次循环后等待1秒再检查状态
                await Task.Delay(1000, cancellationToken);
            }

            // 如果循环结束是因为取消请求，则抛出异常
            cancellationToken.ThrowIfCancellationRequested();

            // 再次等待2秒
            await Task.Delay(2000, cancellationToken);
        }
        private async Task<bool> CmdSendUdp_1Async(PrintCmd cmd, CancellationToken cancellationToken)
        {
            Console.WriteLine("current cmd is: " + cmd.cmd);

            // 拼装发送内容
            RobotHeart rh = new RobotHeart
            {
                point = 1,
                cmd = int.Parse(cmd.cmd.Split(' ')[0]),
                data = new MoveData
                {
                    ID_src = int.Parse(cmd.cmd.Split(' ')[1]),
                    ID_dst = int.Parse(cmd.cmd.Split(' ')[2])
                }
            };
            int retrytimes = 4;
        // 发送UDP消息
            ROBOTRTY:   await SendUdpAsync(rh, robotIP, robotPort); // 假设SendUdpAsync是异步版本的方法
            Console.WriteLine("发送CMD=" + cmd.cmd);

            // 等待1秒后，机械臂进入工作状态，这里做超时处理
            await Task.Delay(1000, cancellationToken);

            // 使用CancellationTokenSource来实现超时
            using (var timeoutCts = new CancellationTokenSource(5000)) // 5秒超时
            using (var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutCts.Token))
            {
                while (true)
                {
                    if (ht != null && ht.rob_sta == 1)
                    {
                        Console.WriteLine("机械臂执行指令中...");
                        break;
                    }

                    // 等待500ms再检查状态
                    await Task.Delay(500, linkedCts.Token);
                    if (linkedCts.IsCancellationRequested)
                    {
                        retrytimes = retrytimes - 1;
                        if (retrytimes>0)
                        {
                            Console.WriteLine("robot start time out! retry!");
                            goto ROBOTRTY;

                        }
                        else
                        {
                            return false;
                        }
                      //  return false;
                    }
                }
    }

    // 异步等待机械臂完成工作
    await WaitRobotWorkAsync(cancellationToken);

    return true;
}

 
        /// </summary>
        private bool CmdSendUdp(PrintCmd cmd)
        {
            Console.WriteLine("current cmd is: " + cmd.cmd.ToString());
            // 1.拼装发送内容
            RobotHeart rh = new RobotHeart
            {
                point = 1
            };
            string[] cmds = cmd.cmd.Split(' ');
            rh.cmd = int.Parse(cmds[0]);
            MoveData md = new MoveData
            {
                ID_src = int.Parse(cmds[1]),
                ID_dst = int.Parse(cmds[2])
            };
            rh.data = md;
            // 0 检查对象状态

            //0 检查对象状态
            // 2.发送
            SendUdp(rh);
            Console.WriteLine("发送CMD=" + cmd.cmd);
            // 等1秒后，机械臂进入工作状态. 这里做超时处理。1+5秒都没有进入工作状态。本次指令发送失败
            Thread.Sleep(1000);
            int cnt = 0;
            while (true)
            {
                cnt++;
                Thread.Sleep(500);
                if (ht != null && ht.rob_sta == 1)
                {
                    Console.WriteLine("机械臂执行指令中...");
                    break;
                }

                if (cnt > 10)
                {
                    Console.WriteLine("robot start time out!");
                    return false;
                }
            }
            // 3.机械臂返回执行完成。再等3秒后返回主线程。
            WaitRobotWork();
            return true;
        }

        /// <summary>
        /// UDP发送
        /// </summary>
        /// <param name="rh"></param>
        private void SendUdp(RobotHeart rh)
        {
            UdpClient udpClient2 = new UdpClient();
            string message = JsonConvert.SerializeObject(rh);
            byte[] data = Encoding.UTF8.GetBytes(message);
            _ = udpClient2.Send(data, data.Length, robotIP, robotPort);
            udpClient2.Close();
        }

        private async Task SendUdpAsync(RobotHeart rh, string robotIp, int robotPort)
        {
            try
            {
                using (UdpClient udpClient = new UdpClient())
                {
                    string message = JsonConvert.SerializeObject(rh);
                    byte[] data = Encoding.UTF8.GetBytes(message);

                    await udpClient.SendAsync(data, data.Length, robotIp, robotPort);
                }
            }
            catch (Exception ex)
            {
                // 记录日志或处理异常
                Console.WriteLine($"An error occurred while sending UDP message: {ex.Message}");
            }
        }

        private void DataToUI()
        {
            // 创建数据模型对象
            List<OptionModel> items = new List<OptionModel>()
                {
                    new OptionModel() { Column1 = "当前配方:", Column2 = "Item 2", Column3 = "Item 3", Column4 = "Item 4" },
                    new OptionModel() { Column1 = "", Column2 = "Item 6", Column3 = "Item 7", Column4 = "Item 8" },
                    new OptionModel() { Column1 = "", Column2 = "Item 10", Column3 = "Item 11", Column4 = "Item 12" },
                    new OptionModel() { Column1 = "", Column2 = "Item 2", Column3 = "Item 3", Column4 = "Item 4" },
                    new OptionModel() { Column1 = "", Column2 = "Item 6", Column3 = "Item 7", Column4 = "Item 8" },
                    new OptionModel() { Column1 = "", Column2 = "Item 10", Column3 = "Item 11", Column4 = "Item 12" }
                };
            lvOption.ItemsSource = items;

            //初始化按钮内容
            Btn_AutoCheck.SetUI("#FFFFFF", "#0080FF", "\\Image\\抽查检查.png", "#0080FF", "自检");
            Btn_Init.SetUI("#FFFFFF", "#0080FF", "\\Image\\抽查检查.png", "#0080FF", "初始化");
            Btn_FlowLoad.SetUI("#FFFFFF", "#0080FF", "\\Image\\抽查检查.png", "#0080FF", "配方载入");

            Btn_run.SetUI("#0080FF", "#0080FF", "\\Image\\安全运行.png", "#FFFFFF", "运行");
            Btn_Recover.SetUI("#00A8FF", "#00A8FF", "\\Image\\安全运行.png", "#FFFFFF", "恢复");
            Btn_Stop.SetUI("#FF6464", "#FF6464", "\\Image\\安全运行.png", "#FFFFFF", "停止");
            Btn_Scram.SetUI("#FF0000", "#FF0000", "\\Image\\安全运行.png", "#FFFFFF", "急停");

            //清洗设备
            dev_clear.SetUI("清洗设备", "正常", "\\Image\\线_一键清空.png");
            //移量设备
            dev_weigh.SetUI("移量设备", "正常", "\\Image\\Group 345.png");

            //硅包反应
            dev_reactivity.SetUI(12, "\\Image\\设备.png", "包硅反应", "正常...", "26.5", "2023.09.19");
            //移液工作站(测试用途)
            dev_pipetting.SetUI(5, "\\Image\\实验分析,试剂,实验,检验,DNA检验,化学.png", "移液工作站(测试用途)", "自检中...", "", "");
            // 机械臂
            dev_robot.SetUI("导轨机器人", "空闲", "\\Image\\机器人.png");

            //配液设备组(1)
            dev_liquid_1.SetUI(6, "\\Image\\设备.png", "配液设备组(1)", "正常...", "", "添加量:10mL", "导轨1");
            dev_liquid_2.SetUI(5, "\\Image\\设备.png", "配液设备组(2)", "正常...", "", "添加量:10mL", "导轨2");
            dev_liquid_3.SetUI(7, "\\Image\\实验室.png", "配液设备组(3)", "正常...", "", "添加量:10mL", "导轨3");

            //MoveRobot();
        }

        private static DispatcherTimer timer;
        private static bool flagLeft = false;
        private void MoveRobot()
        {
            timer = new DispatcherTimer
            {
                Interval = TimeSpan.FromSeconds(7)
            };
            timer.Tick += Timer_Tick;
            timer.Start();
        }
        private void Timer_Tick(object sender, EventArgs e)
        {
            if (flagLeft)
            {
                MoveLeft();
            }
            else
            {
                MoveRight();
            }
            flagLeft = !flagLeft;
        }

        private static Storyboard moveStoryboard;
        private void MoveLeft()
        {
            DoubleAnimation animation = new DoubleAnimation
            {
                To = 0,
                Duration = TimeSpan.FromSeconds(6)
            };

            moveStoryboard = new Storyboard();
            moveStoryboard.Children.Add(animation);
            Storyboard.SetTarget(animation, dev_robot);
            Storyboard.SetTargetProperty(animation, new PropertyPath(Canvas.LeftProperty));
            moveStoryboard.Begin();
        }

        private void MoveRight()
        {
            DoubleAnimation animation = new DoubleAnimation
            {
                By = bd_robot.ActualWidth - dev_robot.ActualWidth, // 向右移动50个单位
                Duration = TimeSpan.FromSeconds(6)
            };

            moveStoryboard = new Storyboard();
            moveStoryboard.Children.Add(animation);
            Storyboard.SetTarget(animation, dev_robot);
            Storyboard.SetTargetProperty(animation, new PropertyPath(Canvas.LeftProperty));
            moveStoryboard.Begin();
        }

        private void StopAnimation()
        {
            if (moveStoryboard != null)
            {
                moveStoryboard.Stop();
            }
        }

        private void Btn_Peiye_Loaded(object sender, RoutedEventArgs e)
        {

        }

        private void Btn_run_Loaded(object sender, RoutedEventArgs e)
        {

        }

        private void Btn_Peiye_Click(object sender, RoutedEventArgs e)
        {
            if (spFY.IsOpen)
            {
                //        byte[] jycmd = { 0xaa, 0x01, 0xf6, 0x00, 0x00, 0x00, 0x03, 0x55 };
                //         spFY.Write(jycmd, 0, jycmd.Length);
                byte[] jycmd = { 0xaa, 0x00, 0xff, 0x00, 0x00, 0x00, 0x00, 0x55 };
                spFY.Write(jycmd, 0, jycmd.Length);
                Console.WriteLine("peiye started!");
            }

        }

        private void start_workstation(int cy)
        {
            if (spFY.IsOpen)
            {
                byte byteValue = (byte)(cy & 0xFF);
                //        byte[] jycmd = { 0xaa, 0x01, 0xf6, 0x00, 0x00, 0x00, 0x03, 0x55 };
                //         spFY.Write(jycmd, 0, jycmd.Length);
                byte[] jycmd = { 0xaa, 0x03, 0x0f, 0x00, 0x00, 0x00, byteValue, 0x55 };
                spFY.Write(jycmd, 0, jycmd.Length);
                Console.WriteLine("workstation started!");
            }
        }

        private void Btn_Workstation_Click(object sender, RoutedEventArgs e)
        {
            // start_workstation(0x02);
            Console.WriteLine("add water");
            if (spFY.IsOpen)
            {
                // AddWater1(6,10);
                addwaterbyID(7, 50);
            }
        }


        public enum WorkstationStatus
        {
            Idle = 0,
            Busy = 1,
            Error = -1
        }


        public class UdpCommunicator : IDisposable
    {
        private UdpClient udpClient;
        private IPEndPoint localEndPoint;
        private IPEndPoint remoteEndPoint;
        private IPEndPoint remoteEndPoint_cmd;
       // private UdpClient udpClient_cmd;
        public int rcpid;

        public UdpCommunicator(string localIp, int localPort, string remoteIp, int remotePort,int remotePort_cmd, int rcpid)
            {
                localEndPoint = new IPEndPoint(IPAddress.Parse(localIp), localPort);
                remoteEndPoint = new IPEndPoint(IPAddress.Parse(remoteIp), remotePort);
                remoteEndPoint_cmd = new IPEndPoint(IPAddress.Parse(remoteIp), remotePort_cmd);
                udpClient = new UdpClient(localEndPoint);
                this.rcpid = rcpid;
            }

            public async Task<JObject> ReceiveJsonAsync(TimeSpan timeout)
        {
            try
            {
                using (var cts = new CancellationTokenSource(timeout))
                {
                    var receiveTask = udpClient.ReceiveAsync();
                    if (await Task.WhenAny(receiveTask, Task.Delay(timeout)) == receiveTask)
                    {
                        var result = await receiveTask;
                        string jsonMessage = Encoding.UTF8.GetString(result.Buffer);
                        Console.WriteLine($"Received JSON: {jsonMessage}");
                        return JObject.Parse(jsonMessage);
                    }
                    else
                    {
                        Console.WriteLine("Receive timed out.");
                        return null;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error receiving JSON: {ex.Message}");
                return null;
            }
        }

        public async Task SendCommandAsync(string command)
        {
            try
            {
                byte[] data = Encoding.UTF8.GetBytes(command);
                await udpClient.SendAsync(data, data.Length, remoteEndPoint_cmd);
                Console.WriteLine($"Sent command: {command}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error sending command: {ex.Message}");
            }
        }

        public void Dispose()
        {
            udpClient?.Dispose();
        }
    }

        public static class RobotID2StationID
        {
            // 私有静态字典用于存储键值对
            private static readonly Dictionary<int, int> _correspondence = new Dictionary<int, int>();

            // 静态构造函数，初始化时填充数据
            static RobotID2StationID()
            {
                List<int> list1 = new List<int> { 6, 7, 8, 9, 12, 13, 14, 15, 16, 17 };
                List<int> list2 = new List<int> { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

                if (list1.Count != list2.Count)
                {
                    throw new InvalidOperationException("两个列表长度不匹配！");
                }

                for (int i = 0; i < list1.Count; i++)
                {
                    _correspondence[list1[i]] = list2[i];
                }
            }

            // 公共静态方法，用于查找给定键对应的值
            public static bool getStationID(int key, out int value)
            {
                return _correspondence.TryGetValue(key, out value);
            }
        }

        public class WorkstationManager
        {
            private readonly UdpCommunicator _communicator;

            public WorkstationManager(UdpCommunicator communicator)
            {
                _communicator = communicator;
            }

            public async Task<bool> ExecuteWorkstationTaskAsync()
            {
                try
                {
                    // 检查初始状态
                    var initialStatus = await CheckInitialStatusAsync(TimeSpan.FromSeconds(3));
                    if (initialStatus != WorkstationStatus.Idle)
                    {
                        Console.WriteLine("Workstation is not idle or no response received.");
                        return false;
                    }

                    // 发送启动命令并确认启动
                    bool startedSuccessfully = await StartWorkstationAsync(TimeSpan.FromSeconds(2), 10);
                    if (!startedSuccessfully)
                    {
                        Console.WriteLine("Failed to start the workstation.");
                        return false;
                    }

                    // 等待任务完成
                    return await MonitorCompletionAsync();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error during communication: {ex.Message}");
                    return false;
                }
            }

            private async Task<WorkstationStatus> CheckInitialStatusAsync(TimeSpan timeout)
            {
                await _communicator.SendCommandAsync("{\"cmd\":1,\"rcp\":"+_communicator.rcpid.ToString()+"}");
                Task.Delay(timeout).Wait();

                var response = await _communicator.ReceiveJsonAsync(timeout);
                return response != null && response.ContainsKey("status")
                    ? (WorkstationStatus)response.Value<int>("status")
                    : WorkstationStatus.Error;
            }

            private async Task<bool> StartWorkstationAsync(TimeSpan timeout, int maxAttempts)
            {
                for (int attempt = 1; attempt <= maxAttempts; attempt++)
                {
                    await _communicator.SendCommandAsync("{\"cmd\":2,\"rcp\":" + _communicator.rcpid.ToString() + "}");

                    var response = await _communicator.ReceiveJsonAsync(timeout);
                    if (response != null && response.Value<int>("status") == (int)WorkstationStatus.Busy)
                    {
                        Console.WriteLine("Station is now busy.");
                        return true;
                    }

                    Console.WriteLine($"Attempt {attempt} failed, retrying...");
                }

                Console.WriteLine("Max attempts reached without success.");
                return false;
            }

            private async Task<bool> MonitorCompletionAsync()
            {
                while (true)
                {
                    var response = await _communicator.ReceiveJsonAsync(TimeSpan.FromSeconds(2));
                    if (response != null && response.Value<int>("status") == (int)WorkstationStatus.Idle)
                    {
                        Console.WriteLine("Station has completed the task.");
                        return true;
                    }

                    Console.WriteLine("Still waiting for station to complete the task...");
                }
            }
        }

        // 示例调用
      

    }
}