------------------------------------------------------------------------
--[[ PerplexityCaptioner ]]--
-- Feedback
-- Computes perplexity for language models
-- For now, only works with SoftmaxTree
------------------------------------------------------------------------
local PerplexityCaptioner, parent = torch.class("dp.PerplexityCaptioner", "dp.Feedback")
PerplexityCaptioner.isPerplexityCaptioner = true

function PerplexityCaptioner:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1],
      "Constructor requires key-value arguments")
   local args, name = xlua.unpack(
      {config},
      'PerplexityCaptioner',
      'Computes perplexity for language models',
      {arg='name', type='string', default='perplexity',
       help='name identifying Feedback in reports'}
   )
   config.name = name
   parent.__init(self, config)
   self._nll = 0
end

function PerplexityCaptioner:setup(config)
   parent.setup(self, config)
   self._mediator:subscribe("doneEpoch", self, "doneEpoch")
end

-- exponential of the mean NLL
function PerplexityCaptioner:perplexity()
   -- divide by number of elements in sequence
   return torch.exp(self._nll / self._n_sample)
end

function PerplexityCaptioner:doneEpoch(report)
   if self._n_sample > 0 and self._verbose then
      print(self._id:toString().." perplexity = "..self:perplexity())
   end
end

function PerplexityCaptioner:add(batch, output, report)
   assert(torch.isTypeOf(batch, 'dp.Batch'), "First argument should be dp.Batch")
   -- table outputs are expected of recurrent neural networks   
   
   
   
   
   
end

function PerplexityCaptioner:_reset()
   self._nll = 0
end

function PerplexityCaptioner:report()
   return {
      [self:name()] = {
         ppl = self._n_sample > 0 and self:perplexity() or 0
      },
      n_sample = self._n_sample
   }
end
