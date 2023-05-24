import numpy as np
import matplotlib.pyplot as plt

reward_agent = np.array(
    [[15.610853292884004, 14.257569241386626, 13.680125492321876, 13.23190903381148, 12.851203058038559, 12.513991034369235, 12.20790638271944, 11.925602611896666, 11.662269155083212, 11.414572823388019, 11.180106964539997, 10.95710911824826, 10.744240950427324, 10.54045520022407, 10.34492731506965, 10.157057059950631, 9.976610525777206, 9.806012049983307, 9.6424430166737, 9.483051034491965, 9.328360006858297, 9.174218128945077, 9.025781487248336, 8.880223570458966, 8.734175169235108, 8.596660791026672, 8.456295866268038, 8.322312795799606, 8.188962936488382, 8.058461692054625, 7.931345478263763], [15.97912154143185, 14.630151753393463, 14.05035803826627, 13.597867776797099, 13.211570859575533, 12.867765702685835, 12.554285699234638, 12.263898855564715, 11.99188705299852, 11.734971973244336, 11.490761149767787, 11.257452227291447, 11.03364487014961, 10.818227860987559, 10.610304420347319, 10.40914392238807, 10.214164709048141, 10.025114117911222, 9.841272903779725, 9.662025557707533, 9.487008313494542, 9.315903337481062, 9.14865930864035, 8.984517343759862, 8.823931055683563, 8.666197749118131, 8.5112810671155, 8.358965668849297, 8.209914109077403, 8.062326822477507, 7.917613470939865], [15.55705979948884, 14.234687491372735, 13.666414456592623, 13.223185708136242, 12.845059369900914, 12.508769935160895, 12.2023597522586, 11.918713893412242, 11.653185164862716, 11.402541102305685, 11.164423062950181, 10.937048947713526, 10.719032064194737, 10.509268172313124, 10.30686561969755, 10.111092155152049, 9.921357983183547, 9.737371961501058, 9.558440575212837, 9.38402394223173, 9.21376861793355, 9.047377311173795, 8.884638150754288, 8.72517412260556, 8.568885535945672, 8.415557357952807, 8.265147726087868, 8.117151024947596, 7.971822540912972, 7.828841572609162, 7.688009980238886], [15.616819558246517, 14.302606254690984, 13.73933347519235, 13.300419145796583, 12.926185881895377, 12.593477736439658, 12.29039812633142, 12.009870399284122, 11.747274798826405, 11.499399379988768, 11.263901244377164, 11.039011315928532, 10.82335352500612, 10.615833480177653, 10.415567776270674, 10.2218318698007, 10.03404330751999, 9.851910332649187, 9.674775244750322, 9.502121902794393, 9.333652709616477, 9.168911267941096, 9.007778702198015, 8.849845690396384, 8.695092326815054, 8.543271511726019, 8.394219356740178, 8.247788162435555, 8.103897446123238, 7.962253601611496, 7.822884011131738], [15.880956464658643, 14.590318697284756, 14.037991025658142, 13.607993430907287, 13.24162793919701, 12.916111426379764, 12.619735961997423, 12.345535214884665, 12.088961371888002, 11.84685401111092, 11.616908068948536, 11.39738496213693, 11.186932401427958, 10.984477590238578, 10.789156725654731, 10.600266679352055, 10.417259028377309, 10.240047376331255, 10.067767907324185, 9.899901745014908, 9.735998092184081, 9.575709893748868, 9.4189052149785, 9.265399294227716, 9.114555296920527, 8.966822480613128, 8.821669272854114, 8.678973200549619, 8.538910663495537, 8.400874182936533, 8.265295047177897]]
)
reward_std = np.array(
    [17.959705893098008, 16.38614656824764, 15.714604543023384, 15.192596147157067, 14.748422447895209, 14.354256732628718, 13.995802296855906, 13.66455038140198, 13.354949379934881, 13.063136632072071, 12.78629372615671, 12.522293477271726, 12.269477998641968, 12.026528040509694, 11.792371988150421, 11.56612563597266, 11.34705024294069, 11.134519661368628, 10.927997262188509, 10.727015080978694, 10.531171183178657, 10.340098660873887, 10.153498328413542, 9.971045339375603, 9.792508501235204, 9.617673879056422, 9.446358029922507, 9.278258826973078, 9.113367120918006, 8.9512797925661, 8.79206723672721]
)
reward_opt = np.array(
    [17.496176187945476, 16.001971969257102, 15.366227048213933, 14.872860196610244, 14.453576329932272, 14.08185157151328, 13.744057431192694, 13.432084219008484, 13.140637199303017, 12.866026286854613, 12.60556851473531, 12.35723113454852, 12.119434704118419, 11.890911893740506, 11.6706376634911, 11.457788807023396, 11.251649892016477, 11.051619845243696, 10.857178967953065, 10.6678999702824, 10.483418962949349, 10.303302132948623, 10.127325432337093, 9.955140572535182, 9.78658999806554, 9.621462894531893, 9.459530540988176, 9.300596929762637, 9.144543344814284, 8.991092832966741, 8.840330399313249]
)

delta_list = np.arange(0, 0.31, 0.01)

reward_agent = np.array(reward_agent)
reward_agent_avg = reward_agent.mean(axis=0)
reward_agent_std = reward_agent.std(axis=0)

reward_std = np.array(reward_std)

plt.plot(delta_list, reward_std, label="classical")
plt.plot(delta_list, reward_opt, label="optimal")
plt.plot(delta_list, reward_agent_avg, label="RFZI")
plt.fill_between(delta_list, reward_agent_avg-reward_agent_std, reward_agent_avg+reward_agent_std, color="C1", alpha=0.1)
plt.xlabel(r"$\delta$")
plt.ylabel(r"$\hat{V}_{\pi}(\delta)$")
plt.legend()
plt.savefig(f"./merge_0.5.png", dpi=200)