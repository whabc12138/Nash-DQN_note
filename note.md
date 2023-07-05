# AC算法
ac算法中，actor net 和 critic net 是两个不同的网络  
**二者的网络结构不同，输入不同，得到的输出不同，作用也不同**  
二者的相同点在于需要通过相同的特征提取层，并不是使用相同的网络结构


## actor network
在a网络中，a网络的输入是当前的状态，输出的是根据当前状态所得到的动作，该动作用于逼近**策略模型Π**  
对于Actor网络而言，其更接近于policy gradients算法，在连续动作中选取合适的动作

## critic network
在c网络中，输入是当前的状态以及从actor network中得到的策略，根据状态和策略，对该策略估计一个V值，**该V值为该策略中每个动作对应的Q值的期望**，用于对actor网络得出的策略进行评估，用于更新actor网络。  
而critic网络自己使用计算出的V值，与该策略所得到的reward，计算TD-error $TD-error = \gamma * V(s') + r - V(s)$，表示当前状态的估计值和下一状态的真实值之间的误差，用于更新c网络的loss。  
以上该td-error的计算来自于MDP过程中，Q值与V值之间的转化，原先的$TD = Q(s,a) -V(s)$， 即使用该状态下采取对应的动作对应的Q值，与该状态的V值之间的差值，用于表示当前状态与预期状态之间的差异  
而利用马尔可夫链后，Q值和V值之间可以相互转化，即$Q(s,a) = V(s') + r$。由此，整个TD-error就可全部使用V值来进行计算。  
对于这一点，我们能够看到，根据MDP，我们将Q值转化成了V值，计算Q值的时候需要动作，而计算V值只需要状态即可。这也是下文中，Nash-DQN一文中，AC网络能够使用同一个网络类的原因。即Critic网络的输入可以不需要动作action，只根据状态state就能够计算Value函数，过程中不需要计算Q值，最终使用TD-error对Critic网络进行更新，利用V值对Actor网络进行更新。


## TD-error
critic网络的目标是最小化TD误差，更新自己的网络参数，用于更好的估计V值。  
actor网络的目标是最大化策略的V值，该V值的来源是critic网络。Critic网络的TD误差对影响到Actor网路的更新方向和速度。

# 对于Nash-DQN
这篇文章中使用了AC网络，但是在形式上和严格意义上的AC网络有一些区别，在于传统的AC网络的A、C网络一般和取两个结构区别较大的网络，而在本文中采取了两个较为近似的网路。同时在本文中，对于每个智能体所采取的action，文章使用了优势函数A来进行替代。

## Advantage net
该网络主要包含了两个部分，一个输入的排列不变层，一个主网络层.  
1. 排列不变层  

    该层的输入是标签不变的特征。将这些特征通过该网络进行归一和转化，将转换出的结果与非不变特征结合，作为主网络的输入。  


2. 主网络  

    主网络根据上面的输入，通过三个隐藏层，输出优势函数的参数$\mu$和$\{P, \Phi \}$ ，利用两个参数来近似的拟合优势函数A

```
self.action_net = PermInvariantQNN(in_invar_dim = self.num_players - 1, non_invar_dim = 3, out_dim = output_dim,num_moments=num_moms).cuda()
```
在这个代码中，他定义的`action_net`实际上就是用于拟合优势函数的Advantage net， 该网络在定义时，给出了输出的维度`out_dim = output_dim`，这里的`output_dim`的默认值为4，表示最终网络输出4个参数。用于表示输出的优势函数的参数$\mu$和$\{P, \Phi \}$

`num_moments`这个参数的意义是用于保证该网络的输出和`Value_net`的输出不同。在神经网络的定义代码中，该参数表示编码器的输出维度和解码器的输入维度。具体值为什么取5，暂且含义不明。


```
    def predict_action(self, states):
        """
        Predicts the parameters of the advantage function of a batch of environmental states
        :param states:    List of environmental state objects
        :return:          List of NashFittedValue objects representing the estimated parameters'
        这个函数predeict action是将当前的状态作为action net的输入，计算出action后，将action的列转换成优势函数的参数
        """
        expanded_states = torch.tensor(self.expand_list(states)).float()
        action_list = self.action_net.forward(invar_input = expanded_states[:,3:].cuda(), non_invar_input = expanded_states[:,0:3].cuda())
        # 这里的变化量就是其余智能体的状态，不变量就是第i个的价格、时间、库存
        # 这里有个问题，该程序使用的action
        
        NFV_list = []
        for i in range(0,len(states)):
            NFV_list.append(NashFittedValues(action_list[i*self.num_players:(i+1)*self.num_players,:]))
        #     这里NFV这个类包含参数：mu：action的第一个动作、c1，c2、c3分别表示了优势函数的三个参数
        # 注意这里没有直接返回动作list，而是将action net计算出的action转化成了优势函数的参数列表，也就是使用优势函数来代替action

        return NFV_list
```
从这段预测action的代码中，我们可以看出action_net的实际输入就是当前的state，`expand_states()`这个函数的主要目的就是将输入的state张量进行拆分，拆成不变量和可变量，不变量就是当前时间、价格、智能体自身库存，可变量就是其余智能体库存。网络的输入还是state。


## Value net
该网络包含了4个隐藏层，输入的是当前的状态，输出的是所有智能体的近似V值。



```
self.value_net = PermInvariantQNN(in_invar_dim = self.num_players - 1, non_invar_dim = 3, out_dim = 1).cuda()
```
首先`Value_net`的定义中，输出维度为1，输出的值表示当前智能体的值函数V，`num_moments`没有赋值，使用默认值1。含义暂且不明。

```
    def predict_value(self, states):
        """
        Predicts the nash value of a batch of environmental states
        :param states:    List of environmental state objects
        :return:          Tensor of estimated nash values of all agents for the batch of states
        计算价值
        """
        expanded_states = torch.tensor(self.expand_list(states)).float()
        values = self.value_net.forward(invar_input = expanded_states[:,3:].cuda(), non_invar_input = expanded_states[:,0:3].cuda())
        return values

```
然后在`predict_value()`这个函数中，我们明显能够看出该函数的结构方法与之前的`predict_action()`这个函数几乎不存在区别，也就是二者使用的实际上就是相同的方法，通过神经网络来拟合参数。*在该方法中，并没有计算动作的Q值，根据系统的状态计算了状态的V值*

## AC方法
这篇文章中使用的两个网络的结构几乎是一样的，二者输出的形式也几乎是一致的，区别在于一个输出的维度是4，一个输出的维度是1，二者表示的内容不同。同时内部网络的结构相似但并不完全相同，也就是使用的还是两个网络。这里最大的问题在于，在计算的过程中，两个网路的输入是相同的，都是输入的当前系统状态。这一点在上文中已经说明了，Critic网络不计算Q值，只计算V值，利用V值来更新网络。  
同时在定义的时候，创建了两个不同的网络实例。两个网络的实例在更新的时候是分开更新的，也就是不会同时更新一个网络的参数。只是作者在写代码的时候，为了简化代码，使用了同一个类，从而使得其看上去有些像只使用了一个网络。
下面来分析一下两个网络的更新过程

### Advantage 更新

```
    def compute_action_Loss(self, state_tuples):
        """
        Computes the loss function for the action/advantage-function network
        """
        
        # Squared Loss penalty for Difference in Coefficients (c1,c2,c3) between agents 
        # to ensure consistency among agents 
        penalty = 25
        
        cur_state_list = [tup[0] for tup in state_tuples]
        action_list = torch.tensor([tup[1] for tup in state_tuples])
        next_state_list = [tup[2] for tup in state_tuples]
        reward_list = torch.tensor([tup[3] for tup in state_tuples]).float()
        
        # Indicator of whether current state is last state or not
        isLastState = np.repeat(np.array([s.t > self.T - 1 for s in next_state_list]).astype(int),self.num_players)
        
        curAct = self.predict_action(cur_state_list)                             # Nash Action Coefficient Estimates of current state
        curVal = self.predict_value(cur_state_list).detach().view(-1).cpu()                    # Nash Value of Current state
        nextVal = self.predict_value(next_state_list).detach().view(-1).cpu().data.numpy()    # Nash Value of Next state
        
        #Makes Matrix of Terminal Values
        expanded_next_states = self.expand_list(next_state_list, norm = False)
        term_list = np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_next_states[:,1],expanded_next_states[:,2],self.term_costs*np.ones(len(expanded_next_states)))))
        # 这里是计算当前状态的终止值

        # Create Lists for predicted Values
        c1_list = torch.stack([nfv.c1 for nfv in curAct]).view(-1).cpu()
        c2_list = torch.stack([nfv.c2 for nfv in curAct]).view(-1).cpu()
        c3_list = torch.stack([nfv.c3 for nfv in curAct]).view(-1).cpu()
        mu_list = torch.stack([nfv.mu for nfv in curAct]).cpu()
        # 这里是从action中将优势函数的参数提取出来

        #Creates the Mu_neg and u_Neg Matrices
        uNeg_list = torch.tensor(self.matrix_slice(action_list)).float().cpu()
        muNeg_list = self.matrix_slice(mu_list).cpu()
        act_list = torch.tensor(action_list.view(-1)).float().cpu()
        mu_list = mu_list.view(-1).cpu()

        #Computes the Advantage Function using matrix operations
        A = - c1_list * (act_list-mu_list)**2 / 2 - c2_list * (act_list-mu_list) * torch.sum(uNeg_list - muNeg_list,
                        dim = 1) - c3_list * torch.sum((uNeg_list - muNeg_list)**2,dim = 1) / 2
        # 计算优势函数


        return torch.sum((torch.tensor(np.multiply(np.ones(len(curVal))-isLastState, nextVal) + np.multiply(isLastState, term_list) + reward_list.view(-1)).float()
                          - curVal - A)**2 
                        + penalty*torch.var(c1_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1)
                        + penalty*torch.var(c2_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1)
                        + penalty*torch.var(c3_list.view(-1,self.num_players),1).view(-1,1).repeat(1,self.num_players).view(-1)).cuda()
        # 根据优势函数（action），当前价值，下一状态价值，奖励一起计算了action loss
```

这一串我们能够看到`predict_action()`计算出的是优势函数的参数列表，在下面定义的四个list分别接收了这些参数，`curVal`表示当前状态对应的V值，也就是$V(s)$ ，`nextVal` 表示下一个状态的$V(s')$， `reward_list`用于存放当前动作的奖励r。通过这四者，即使用了Critic网络，用该网路的输出V，来对Actor网络的更新做出了参考。能够看到最后该函数返回的`action_loss`中，是使用了两个V值来对Actor网络来进行更新。这里使用的是优势函数的方法，代替了上文中介绍AC算法中使用TD-error的方法，但是本质是一样的。

### Value 更新

```
    def compute_value_Loss(self,state_tuples):
        """
        Computes the loss function for the value network
        :param state_tuples:    List of a batch of transition tuples
        :return:                Total loss of batch
        """
        cur_state_list = [tup[0] for tup in state_tuples]
        next_state_list = [tup[2] for tup in state_tuples]
        reward_list = [tup[3] for tup in state_tuples]
        
        # Indicator of whether current state is last state or not
        isLastState = np.repeat(np.array([s.t > self.T - 1 for s in next_state_list]).astype(int),self.num_players)
        #  判断是否达到了最终状态
        
        target = self.predict_value(cur_state_list).view(-1)
        expanded_states = self.expand_list(cur_state_list, norm = False)
        expanded_next_states = self.expand_list(next_state_list, norm = False)
        # 计算了当前状态的V值


        term_list = np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_states[:,1],expanded_states[:,2]/2,self.transaction_cost*np.ones(len(expanded_states))))) \
                    + np.array(list(map(lambda p,q,tc: q*p - tc*q**2, expanded_next_states[:,1],expanded_states[:,2]/2,self.transaction_cost*np.ones(len(expanded_states)))))
        nextstate_val = self.predict_value(next_state_list).detach().view(-1).data.cpu().numpy()
        # 计算下一状态的V值

        # Returns the squared loss with target being:
        # Sum of current state's reward + estimated next state's nash value     --- if not last state
        # Reward obtained if agent executes action to net out current inventory --- if last state
        return self.criterion(target,
                              torch.tensor(np.multiply(isLastState,term_list) + np.multiply(np.ones(len(expanded_states)) - isLastState,
                              nextstate_val + np.array(reward_list).flatten())).float().cuda())
        # 根据当前状态V值、下一状态V值、奖励计算Vloss
```
同样的近似的来看这个预测vloss的函数，该函数也能看到计算了当前状态的$V(s)$以及下一状态的$V(s')$，以及当前状态的奖励r。根据这三者来计算了Value_net的损失函数。因为计算的是V值，因此不需要输入动作action，就能够完成计算。

# 总结

AC算法中，需要一个Actor网络和一个Critic网络，Actor网络输入当前环境状态，输出当前动作。Critic网络输入当前动作和当前环境状态，输出当前状态对应的V值。Actor网络和Critic网络的结构之间没有必然联系，网络的输入不同，输出的目标也不同。
Actor网络的更新目标是最大化自己选择的动作的V值，更新的根据来源是Critic网络的输出。  
Critic网络的更新目标是最小化自己的TD-error，利用自身计算的当前状态V值和目标状态V值来进行更新。  
  
对于该文章而言，文章对DQN的计算是基于V值的，而不是Q值。因此两个网络的输入可以是一样的，都输入的是系统的state，根据state，两个网络分别计算了优势函数A的参数，以及state对应的Value值。根据V值，对Actor网络以及value网络进行更新。  

如果说想要利用Q值来进行计算，这时就需要严格定义两个不同的网络，Actor输入接受状态，Critic输入接受动作和状态，由此Critic能够计算动作对应的Q值，根据Q值对两个网络进行更新。